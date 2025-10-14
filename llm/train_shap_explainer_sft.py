import argparse, json, os, math, random
from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Optional: bitsandbytes for QLoRA
try:
    from transformers import BitsAndBytesConfig
    BNB_AVAILABLE = True
except Exception:
    BNB_AVAILABLE = False

# PEFT / LoRA
from peft import LoraConfig, get_peft_model
try:
    from peft import prepare_model_for_kbit_training
except Exception:
    def prepare_model_for_kbit_training(m, use_gradient_checkpointing=True):
        # Fallback no-op if peft version lacks this helper
        if use_gradient_checkpointing:
            m.gradient_checkpointing_enable()
            m.enable_input_require_grads()
        return m

SYSTEM_PROMPT = ""
PROMPT_PREFIX = "Instruction:\n"
ANSWER_PREFIX = "\n\nAnswer:\n"

class ShapJsonlDataset(Dataset):
    def __init__(self, path: str, tokenizer: AutoTokenizer, max_len: int = 1024):
        self.rows = []
        self.tok = tokenizer
        self.max_len = max_len
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                instr = obj.get("instruction", "")
                resp = obj.get("response", "")
                self.rows.append((instr, resp))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        instr, resp = self.rows[idx]
        prompt = f"{PROMPT_PREFIX}{instr}{ANSWER_PREFIX}"
        prompt_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        resp_ids = self.tok(resp + self.tok.eos_token, add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + resp_ids
        labels = [-100] * len(prompt_ids) + resp_ids[:]

        if len(input_ids) > self.max_len:
            input_ids = input_ids[-self.max_len:]
            labels = labels[-self.max_len:]

        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

@dataclass
class LMDataCollator:
    tokenizer: AutoTokenizer
    pad_to_multiple_of: int = 8

    def __call__(self, features: List[Dict]):
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            max_len = int(math.ceil(max_len / self.pad_to_multiple_of) * self.pad_to_multiple_of)
        def pad(t, value): return torch.nn.functional.pad(t, (0, max_len - len(t)), value=value)
        return {
            "input_ids": torch.stack([pad(f["input_ids"], self.tokenizer.pad_token_id) for f in features]),
            "attention_mask": torch.stack([pad(f["attention_mask"], 0) for f in features]),
            "labels": torch.stack([pad(f["labels"], -100) for f in features]),
        }

class BoosterTrainer(Trainer):
    def __init__(self, *args, loss_booster_epochs: int = 0, boost_factor: float = 1.15, **kwargs):
        # tokenizer arg deprecation handled via processing_class below
        super().__init__(*args, **kwargs)
        self.loss_booster_epochs = loss_booster_epochs
        self.boost_factor = boost_factor

    # Accept **kwargs for HF's evolving signature (e.g., num_items_in_batch)
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss
        if self.loss_booster_epochs and self.state.epoch is not None:
            if self.state.epoch < float(self.loss_booster_epochs):
                loss = loss * self.boost_factor
        return (loss, outputs) if return_outputs else loss

def print_trainable_parameters(model):
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        numel = p.numel()
        total += numel
        if p.requires_grad:
            trainable += numel
    pct = 100 * trainable / total if total else 0.0
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss_booster_epochs", type=int, default=0)
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    # LoRA / QLoRA controls
    parser.add_argument("--use_lora", type=int, default=1)
    parser.add_argument("--qlora", type=int, default=0)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed); torch.manual_seed(args.seed)

    print(f"Loading model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # dtype for compute
    use_cuda = torch.cuda.is_available()
    bf16_ok = use_cuda and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16_ok else torch.float16

    # Model load: 4-bit if --qlora, else standard
    quant_config = None
    model_kwargs = {"device_map": "auto"}
    if args.qlora:
        if not BNB_AVAILABLE:
            raise RuntimeError("bitsandbytes not available. Install it or run without --qlora.")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, quantization_config=quant_config, device_map="auto"
        )
    else:
        model_kwargs["torch_dtype"] = compute_dtype  # safe here
        model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)

    # Gradient checkpointing + disable cache
    gradient_checkpointing = True
    if gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    # LoRA (default on). This slashes VRAM vs full fine-tune.
    if args.use_lora:
        lora_targets = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
        if args.qlora:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        lora_cfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            bias="none", task_type="CAUSAL_LM", target_modules=lora_targets
        )
        model = get_peft_model(model, lora_cfg)
        print_trainable_parameters(model)

    train_ds = ShapJsonlDataset(args.data_path, tokenizer, max_len=args.max_len)
    print(f"Loaded {len(train_ds)} training examples from {args.data_path}")

    collator = LMDataCollator(tokenizer)
    fp16_flag = use_cuda and not bf16_ok

    # Optimizer: use paged 8-bit Adam when QLoRA, else AdamW
    optim_name = "paged_adamw_8bit" if args.qlora else "adamw_torch"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        optim=optim_name,
        weight_decay=0.0,
        fp16=fp16_flag,
        bf16=(not fp16_flag) and use_cuda,
        gradient_checkpointing=gradient_checkpointing,
        report_to="none",
    )

    trainer = BoosterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
        processing_class=tokenizer,  # future-proof vs tokenizer=
        loss_booster_epochs=args.loss_booster_epochs,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved fine-tuned model to {args.output_dir}")

if __name__ == "__main__":
    # Fragmentation helper on Windows if you want (set before running):
    # os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()

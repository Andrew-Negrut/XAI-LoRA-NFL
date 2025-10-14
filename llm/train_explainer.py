from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import os

MODEL_ID = os.environ.get("MODEL_ID", "meta-llama/Meta-Llama-3.1-8B-Instruct")
OUT_DIR  = "models/llama_explainer"

# 1) Load base + tokenizer (keep it simple on Windows: no bitsandbytes here)
tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"
tok.model_max_length = 512      # <-- enforce max length here (replaces max_seq_length)

# If you want to avoid any CPU offload, force a single-GPU map:
# device_map={"":0} keeps everything on cuda:0 (needs enough VRAM)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map={"": 0},         # was "auto"; change back if you prefer auto-sharding
    torch_dtype="auto",
)

# 2) Add LoRA
peft_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_cfg)

# 3) Load data and build a single 'text' per row
raw = load_dataset("json", data_files="llm/explain_train.jsonl")

def to_text(ex):
    # Match your inference prompt structure
    return {
        "text": (
            "You are an explainer. Be concise and factual.\n\n"
            "### Input (JSON):\n" + ex["input"] + "\n\n### Response:\n" + ex["output"]
        )
    }

ds = raw["train"].map(to_text)
split = ds.train_test_split(test_size=0.1, seed=42)

# 4) SFT config (trl==0.23.0) — remove max_seq_length here
cfg = SFTConfig(
    output_dir=OUT_DIR,
    num_train_epochs=2,                 # bump to 3–4 if needed
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    logging_steps=10,
    eval_strategy="epoch",              # correct key for your version
    save_strategy="epoch",
    bf16=True,                          # your 5090 supports bf16
    dataset_text_field="text",          # <— important
    packing=False,                      # one sample per sequence
    report_to="none",
)

# 5) Trainer — in trl 0.23 use 'processing_class' instead of 'tokenizer'
trainer = SFTTrainer(
    model=model,
    args=cfg,
    train_dataset=split["train"],
    eval_dataset=split["test"],
    processing_class=tok,               # <— was tokenizer=tok
)

trainer.train()

# 6) Save a clean, final adapter (no checkpoint hunting later)
trainer.model.save_pretrained(f"{OUT_DIR}/final")
tok.save_pretrained(f"{OUT_DIR}/final")
print("Saved:", f"{OUT_DIR}/final")

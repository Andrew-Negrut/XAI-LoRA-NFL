import os, json, random, argparse, re
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from joblib import load
import shap

from transformers import AutoTokenizer, AutoModelForCausalLM

# ================= Config (env overrides) =================
MODEL_ID = os.environ.get("NATURAL_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
DEVICE_MAP = "auto"

TEMP = float(os.environ.get("GEN_TEMP", "0.9"))
TOP_P = float(os.environ.get("TOP_P", "0.97"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "220"))
MIN_NEW_TOKENS = int(os.environ.get("MIN_NEW_TOKENS", "40"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "24"))
MAX_GENERATION_PASSES = int(os.environ.get("MAX_GENERATION_PASSES", "8"))

NUM_TOL = float(os.environ.get("NUM_TOL", "0.20"))  # a bit looser
ALIGN_WITH_OUTCOME = os.environ.get("ALIGN_WITH_OUTCOME", "1") == "1"
RNG_SEED = int(os.environ.get("SEED", "42"))

DEFAULT_OUT = "data/shap_explainer.jsonl"
DEFAULT_REJ = "data/shap_explainer_rejects.jsonl"

# ================= Small utils =================
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def round2(x: float) -> float:
    return float(f"{x:.2f}")

def as_pct(p: float) -> int:
    return int(round(max(0.0, min(1.0, float(p))) * 100))

def label_str(y: int) -> str:
    return "Home win" if int(y) == 1 else "Home loss"

def normalize_text(s: str) -> str:
    # remove non-alnum, compress spaces, lowercase
    s = re.sub(r"[^A-Za-z0-9]+", " ", s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

# ================= Verification helpers =================
POS_CUES = r"(help|support|boost|favor|aid|assist|contribute[d]?\s+positively|pushed\s+(?:toward|to)\s+(?:a\s+)?win)"
NEG_CUES = r"(hurt|harm|drag|worsen|undermine|penalize|contribute[d]?\s+negatively|pushed\s+(?:toward|to)\s+(?:a\s+)?loss)"

def _extract_two_sentences_anywhere(text: str) -> Tuple[str, str]:
    """
    Find the sentence that starts with 'Prediction:' (case-insensitive),
    then take the very next sentence as s2.
    """
    # split to sentences
    parts = re.split(r'(?<=[\.!?])\s+', text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    # find first 'Prediction:' sentence
    idx = -1
    for i, p in enumerate(parts):
        if p.lower().startswith("prediction:"):
            idx = i; break
    if idx == -1:
        # fallback: try to locate 'Prediction:' substring inside a sentence
        for i, p in enumerate(parts):
            if "prediction:" in p.lower():
                idx = i; break
    if idx == -1:
        return text.strip(), ""
    s1 = parts[idx]
    s2 = parts[idx+1] if idx+1 < len(parts) else ""
    return s1, s2

def _name_present(text: str, feature_name: str) -> bool:
    # accept minor formatting drift: hyphens/spaces/case removed
    norm_text = normalize_text(text)
    norm_name = normalize_text(feature_name)
    return norm_name in norm_text

def _window_after(text: str, anchor: str, span: int = 180) -> str:
    # get small window after the first occurrence of anchor (normalized)
    t = text
    m = re.search(re.escape(anchor), t, flags=re.I)
    if not m:
        # try normalized
        nt = normalize_text(t); na = normalize_text(anchor)
        pos = nt.find(na)
        if pos == -1:
            return ""
        # can't map back exactly; return a wide window of original text
        return t[:][: span * 2]  # coarse fallback window
    start = m.end()
    return t[start : start + span]

def _has_number_near(window: str) -> bool:
    return re.search(r"[-+]?\d+(?:\.\d+)?", window) is not None

def _extract_shap_vals(window: str) -> List[float]:
    vals = []
    # Accept SHAP +2, +2.3, +2.30 (1-3 decimals)
    for m in re.finditer(r"SHAP\s*([+\-])\s*(\d+(?:\.\d{1,3})?)", window, re.I):
        sgn = -1.0 if m.group(1) == "-" else 1.0
        mag = float(m.group(2))
        vals.append(sgn * mag)
    return vals

def _has_shap_value(window: str, expected: float, tol: float = NUM_TOL) -> bool:
    vals = _extract_shap_vals(window)
    if not vals:
        return False
    return any(abs(v - expected) <= tol for v in vals)

def _has_consistent_cue(window: str, sign: int) -> bool:
    """Check that the window has the correct cue AND does not have contradictory cues."""
    has_pos = re.search(POS_CUES, window, re.I) is not None
    has_neg = re.search(NEG_CUES, window, re.I) is not None
    
    if sign > 0:
        # Must have positive cue, must NOT have negative cue
        return has_pos and not has_neg
    if sign < 0:
        # Must have negative cue, must NOT have positive cue
        return has_neg and not has_pos
    return True

def _pct_from_text(s1: str) -> int | None:
    # find first percentage like 73% or 73.0%
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", s1)
    if not m:
        return None
    val = float(m.group(1))
    return int(round(val))

def accept(s1: str, s2: str, proba: float, drivers: List[Dict]) -> bool:
    if not s1 or not s2:
        return False

    # s1 must communicate "Prediction: Home win/loss (NN%)"
    if "prediction:" not in s1.lower():
        return False
    pct_true = as_pct(proba)
    pct_found = _pct_from_text(s1)
    if pct_found is None or abs(pct_found - pct_true) > 1:
        return False

    # s2 must reference SHAP and not be obviously broken
    if "shap" not in s2.lower():
        return False
    if re.search(r"\bSHAP\s+of\s+SHAP\b", s2, re.I):
        return False

    outcome_cue_ok = False
    for d in drivers:
        name = d["name"]
        exp = round2(float(d["contribution"]))
        sign = 1 if exp > 0 else (-1 if exp < 0 else 0)

        if not _name_present(s2, name):
            return False
        win = _window_after(s2, name, span=120)
        if not win:
            return False
        if not _has_number_near(win):
            return False
        if not _has_shap_value(win, exp, tol=NUM_TOL):
            return False
        if not _has_consistent_cue(win, sign):
            return False

        if sign > 0 and re.search(POS_CUES, win, re.I):
            outcome_cue_ok = True
        if sign < 0 and re.search(NEG_CUES, win, re.I):
            outcome_cue_ok = True

    if ALIGN_WITH_OUTCOME and not outcome_cue_ok:
        return False

    return True

# ================= Feature space =================
FEATURES = [
    "yards_per_play_diff", "turnovers_diff", "passing_yards",
    "rushing_yards", "sacks", "penalties", "third_down_rate"
]
DISPLAY = {
    "yards_per_play_diff": "Yards per play differential",
    "turnovers_diff": "Turnover differential",
    "passing_yards": "Passing yards",
    "rushing_yards": "Rushing yards",
    "sacks": "Sacks taken",
    "penalties": "Penalties committed",
    "third_down_rate": "Third-down conversion rate",
}

def sample_row() -> Dict:
    return {
        "yards_per_play_diff": round(random.uniform(-1.5, 1.8), 2),
        "turnovers_diff": round(random.uniform(-2.5, 2.5), 2),
        "passing_yards": float(int(random.uniform(120, 420))),
        "rushing_yards": float(int(random.uniform(50, 220))),
        "sacks": int(random.uniform(0, 7)),
        "penalties": int(random.uniform(0, 12)),
        "third_down_rate": round(random.uniform(0.25, 0.55), 3),
    }

# ================= LLM =================
def load_llm(model_id: str = MODEL_ID):
    print(f"[builder] Loading LLM: {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=DEVICE_MAP,
        torch_dtype=dtype,
    )
    model.eval()

    if hasattr(model, "config"):
        model.config.pad_token_id = tok.pad_token_id
        model.config.eos_token_id = tok.eos_token_id
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tok.pad_token_id
        model.generation_config.eos_token_id = tok.eos_token_id

    return tok, model

def build_prompt(pred_label: int, proba: float, top: List[Dict]) -> str:
    pct = as_pct(proba)
    # Force exact names by quoting them
    lines = []
    for d in top:
        name = d["name"]
        val = d["value"]
        contrib = round2(float(d["contribution"]))
        sign_word = "helped" if contrib > 0 else ("hurt" if contrib < 0 else "neutral")
        lines.append(f'- "{name}": value={val}, SHAP={contrib:+.2f} → {sign_word}')

    instruction = (
        "Write exactly two sentences.\n"
        f"Sentence 1 must be exactly: 'Prediction: {label_str(pred_label)} ({pct}%).'\n"
        "Sentence 2 must mention each feature by its exact quoted name, state its raw value, then say 'SHAP ±X.XX' with the exact number shown below.\n"
        "CRITICAL: If SHAP is positive (SHAP +X.XX), you MUST say it 'helped' or 'boosted' the prediction. "
        "If SHAP is negative (SHAP -X.XX), you MUST say it 'hurt' or 'reduced' the prediction. "
        "The SHAP sign determines helped/hurt, NOT the raw value. Never contradict the SHAP sign."
    )
    bullet = "\n".join(lines)
    user = (
        f"{instruction}\n\n"
        "Features (use names exactly as quoted):\n"
        f"{bullet}\n\n"
        "Remember: SHAP sign = helped/hurt direction. Use the exact numbers shown."
    )
    return user

def make_chat_prompt(tok, user_msg: str) -> str:
    messages = [
        {"role": "system", "content": "You are a concise sports analyst that explains model predictions clearly and factually."},
        {"role": "user", "content": user_msg},
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def generate_batch(tok, model, prompts: List[str]) -> List[str]:
    chat_prompts = [make_chat_prompt(tok, p) for p in prompts]
    toks = tok(chat_prompts, return_tensors="pt", padding=True, truncation=True)
    toks = {k: v.to(model.device) for k, v in toks.items()}

    with torch.no_grad():
        out = model.generate(
            **toks,
            do_sample=True,
            temperature=TEMP,
            top_p=TOP_P,
            max_new_tokens=MAX_NEW_TOKENS,
            min_new_tokens=MIN_NEW_TOKENS,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    decoded = tok.batch_decode(out, skip_special_tokens=True)

    results = []
    for i, full in enumerate(decoded):
        txt = full
        # Extract only the assistant's response by finding the last "assistant" marker
        if "assistant" in txt.lower():
            # Split on "assistant" (case-insensitive) and take everything after the last occurrence
            parts = re.split(r'\bassistant\b', txt, flags=re.I)
            if len(parts) > 1:
                txt = parts[-1].lstrip()
        results.append(txt.strip())
    return results

# ================= Synthesis + SHAP =================
def build_rows(n_rows: int, xgb, FEATURES: List[str], explainer) -> List[Dict]:
    rows = []
    for _ in range(n_rows):
        row = sample_row()
        X = pd.DataFrame([row])[FEATURES]
        proba = float(xgb.predict_proba(X)[:, 1][0])
        y = int(proba >= 0.5)

        sv = explainer.shap_values(X)[0]
        idx = np.argsort(-np.abs(sv))[:3]
        top = []
        for i in idx:
            feat = FEATURES[i]
            top.append({
                "feature": feat,
                "name": DISPLAY.get(feat, feat),
                "value": float(row[feat]),
                "contribution": round2(float(sv[i])),
                "direction": "positive" if sv[i] > 0 else ("negative" if sv[i] < 0 else "neutral"),
            })
        rows.append({"row": row, "proba": proba, "label": y, "top": top})
    return rows

# ================= Re-verification =================
def reverify_accepted(input_path: str, output_path: str, rejects_path: str):
    """Re-verify an existing accepted dataset and filter out bad entries."""
    print(f"[reverify] Loading {input_path}")
    accepted = []
    rejected = []
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            
            # Parse instruction to get drivers
            inst = json.loads(entry["instruction"])
            proba = float(inst["probability"])
            drivers = inst["top_features"]
            
            # Extract sentences from response
            text = entry["response"]
            s1, s2 = _extract_two_sentences_anywhere(text)
            
            # Re-verify
            if accept(s1, s2, proba, drivers):
                accepted.append(entry)
            else:
                rejected.append(entry)
    
    # Write cleaned accepted
    print(f"[reverify] Writing {len(accepted)} clean entries to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in accepted:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    # Write filtered rejects
    if rejected:
        print(f"[reverify] Writing {len(rejected)} filtered entries to {rejects_path}")
        with open(rejects_path, "w", encoding="utf-8") as f:
            for entry in rejected:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"\n[reverify] Summary: {len(accepted)} accepted, {len(rejected)} rejected")
    return len(accepted), len(rejected)

# ================= Main =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_rows", type=int, default=800)
    parser.add_argument("--out", type=str, default=DEFAULT_OUT)
    parser.add_argument("--rejects", type=str, default=DEFAULT_REJ)
    parser.add_argument("--reverify", action="store_true", help="Re-verify existing accepted file")
    args = parser.parse_args()

    set_seed(RNG_SEED)
    out_path = Path(args.out); rej_path = Path(args.rejects)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # If reverify mode, clean existing accepted file and exit
    if args.reverify:
        if not out_path.exists():
            print(f"[error] File not found: {out_path}")
            return
        reverify_accepted(str(out_path), str(out_path), str(rej_path.parent / "shap_explainer_filtered.jsonl"))
        return

    # Classic model + SHAP
    xgb, feature_list = load("models/xgb_homewin.joblib")
    features = feature_list if isinstance(feature_list, list) else FEATURES
    explainer = shap.TreeExplainer(xgb)

    print(f"[builder] Creating {args.n_rows} synthetic rows…")
    base = build_rows(args.n_rows, xgb, features, explainer)

    tok, model = load_llm(MODEL_ID)

    accepted = []
    remaining = list(range(len(base)))
    last_reply: Dict[int, str] = {}
    prompts_cache: Dict[int, str] = {}

    for gen_pass in range(1, MAX_GENERATION_PASSES + 1):
        if not remaining: break
        print(f"[gen pass {gen_pass}/{MAX_GENERATION_PASSES}] pending: {len(remaining)}")

        prompts = []
        metas = []
        for idx in remaining:
            item = base[idx]
            prompt = build_prompt(item["label"], item["proba"], item["top"])
            prompts.append(prompt); metas.append(idx)
            prompts_cache[idx] = prompt

        batch_texts = []
        for i in tqdm(range(0, len(prompts), BATCH_SIZE)):
            chunk = prompts[i:i+BATCH_SIZE]
            outs = generate_batch(tok, model, chunk)
            batch_texts.extend(outs)

        still_pending = []
        for meta_idx, text in zip(metas, batch_texts):
            last_reply[meta_idx] = text
            s1, s2 = _extract_two_sentences_anywhere(text)
            item = base[meta_idx]
            ok = accept(s1, s2, item["proba"], item["top"])
            if ok:
                accepted.append({
                    "instruction": json.dumps({
                        "prediction": item["label"],
                        "probability": float(item["proba"]),
                        "top_features": item["top"],
                    }),
                    "response": f"{s1.strip()} {s2.strip()}",
                })
            else:
                still_pending.append(meta_idx)

        remaining = still_pending
        print(f"  -> pass {gen_pass} accepted: {len(accepted)} / {args.n_rows} (pending {len(remaining)})")

    # Write accepted
    print(f"\n[builder] Writing accepted to {out_path}")
    with out_path.open("w", encoding="utf-8") as f:
        for ex in accepted:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Write rejects with last reply for inspection
    print(f"[builder] Writing {len(remaining)} rejects to {rej_path}")
    with rej_path.open("w", encoding="utf-8") as f:
        for idx in remaining:
            item = base[idx]
            f.write(json.dumps({
                "instruction": json.dumps({
                    "prediction": item["label"],
                    "probability": float(item["proba"]),
                    "top_features": item["top"],
                }),
                "prompt": prompts_cache.get(idx, ""),
                "response": last_reply.get(idx, "")
            }, ensure_ascii=False) + "\n")

    nat = len(accepted); tot = len(base)
    print("\n================ SUMMARY ================")
    print(f"Natural written       : {nat}")
    print(f"Total                 : {tot}")
    if tot > 0:
        print(f"Natural : Total = {nat} / {tot} = {nat/tot:.2f}")
    print("========================================")

if __name__ == "__main__":
    main()

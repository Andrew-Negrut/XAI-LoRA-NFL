import json
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
from joblib import load
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import shap

# =========================
# Friendly display labels
# =========================
FRIENDLY = {
    "yards_per_play_diff": "Yards per play differential",
    "turnovers_diff": "Turnover differential",
    "passing_yards": "Passing yards",
    "rushing_yards": "Rushing yards",
    "sacks": "Sacks taken",
    "penalties": "Penalties committed",
    "third_down_rate": "Third-down conversion rate",
}

# =========================
# Style controls (constrained variety)
# =========================
PHRASE_POS = ["helped", "supported", "boosted", "contributed positively"]
PHRASE_NEG = ["hurt", "weighed against", "dragged down", "detracted"]
CONNECTORS = ["driven by", "supported by", "with", "while", "and", "despite", "tempered by", "backed by"]
BANNED_WORDS = [
    "more", "less", "most", "least", "bigger", "smaller", "stronger", "weaker",
    "largest", "smallest", "top", "dominant", "biggest", "highest", "lowest"
]

# Debug: set DEBUG_XAI=1 to print raw reply & verifier notes
DEBUG_XAI = os.environ.get("DEBUG_XAI", "0") == "1"

# Optional sampling controls (env)
XAI_VARIETY = os.environ.get("XAI_VARIETY", "off").lower()  # off | stable | random
TEMP = float(os.environ.get("TEMP", "0.4"))
TOP_P = float(os.environ.get("TOP_P", "0.92"))
TOP_K = int(os.environ.get("TOP_K", "40"))
REP_PEN = float(os.environ.get("REP_PEN", "1.05"))
# Auto-retry attempts (new)
XAI_RETRIES = int(os.environ.get("XAI_RETRIES", "3"))

# =========================
# Load tabular model
# =========================
xgb, FEATURES = load("models/xgb_homewin.joblib")  # (model, feature_list)

# =========================
# Load base LLM + LoRA
# =========================
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
ADAPTER_DIR_ENV = os.environ.get("ADAPTER_DIR")  # e.g., models\llama_explainer\qwen_sft_v1

base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    dtype=torch.bfloat16 if torch.cuda.is_available() else None,
)
tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.padding_side = "right"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
base.config.pad_token_id = tok.pad_token_id
base.config.eos_token_id = tok.eos_token_id
base.generation_config.pad_token_id = tok.pad_token_id
base.generation_config.eos_token_id = tok.eos_token_id

def _find_latest_adapter(root: str) -> str:
    rp = Path(root)
    if not rp.exists():
        raise FileNotFoundError(f"Adapter root not found: {root}")
    cands = []
    for p in rp.iterdir():
        if p.is_dir() and (p / "adapter_config.json").exists():
            step = 10**12 if p.name == "final" else -1
            if p.name.startswith("checkpoint-"):
                try:
                    step = int(p.name.split("-")[-1])
                except Exception:
                    step = -1
            cands.append((step, p))
    if not cands:
        raise FileNotFoundError(f"No adapter_config.json under {root}")
    cands.sort(key=lambda x: x[0])
    return str(cands[-1][1])

adapter_dir = ADAPTER_DIR_ENV or _find_latest_adapter("./models/llama_explainer")
print(f"[pipeline] Loading LoRA from: {adapter_dir}")
model = PeftModel.from_pretrained(base, adapter_dir)
model.eval()

# =========================
# SHAP explainer
# =========================
explainer = shap.TreeExplainer(xgb)

# =========================
# Helper: fixed-format strings for values
# =========================
def _fmt_val(v):
    if isinstance(v, float):
        s = f"{v:.2f}"
        s = s.rstrip("0").rstrip(".")
        return s
    return str(int(v))

# =========================
# Prompt builder (constrained variety)
# =========================
def _build_messages(payload, allowed_names):
    names_list = ", ".join([f"'{n}'" for n in allowed_names])
    style_rules = (
        "Write exactly two sentences.\n"
        "Sentence 1 MUST be exactly: 'Prediction: <Home win|Home loss> (<XX%>).'\n"
        "Sentence 2 MUST be a single flowing sentence that mentions EACH driver using the EXACT names "
        f"{names_list} (match capitalization). For each driver, include exactly the strings: "
        "value <value_str> and SHAP <shap_str> (use them verbatim). "
        "You may place SHAP in parentheses or after a comma.\n"
        "Use natural connectors chosen only from: " + ", ".join(CONNECTORS) + ". "
        "For positive SHAP, use exactly one verb chosen only from: " + ", ".join(PHRASE_POS) + "; "
        "for negative SHAP, from: " + ", ".join(PHRASE_NEG) + ". "
        "Avoid ANY comparative/superlative wording and DO NOT use these words: "
        + ", ".join(BANNED_WORDS) + ". "
        "Do not invent numbers. Do not use bullet points. Keep it concise."
    )
    return [
        {
            "role": "system",
            "content": "You explain model predictions to non-technical users. Be concise, factual, and specific. Use the numbers given."
        },
        {"role": "user", "content": style_rules + "\n\n" + json.dumps(payload, ensure_ascii=False)},
    ]

# =========================
# Helper: extract the assistant reply text
# =========================
def _extract_assistant_reply(decoded_text: str, prompt_text: str) -> str:
    text = decoded_text.strip()
    if prompt_text and text.startswith(prompt_text):
        text = text[len(prompt_text):].lstrip()
    low = text.lower()
    if "assistant" in low:
        text = text.split("assistant", 1)[-1].strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)[:1200].strip()

def _make_value_pattern(val_str: str) -> str:
    """
    Build a regex that matches the exact numeric value but tolerates:
      - optional space after sign: - 2.5
      - optional leading zero for decimals <1: .22 or 0.22
      - optional trailing zeros: 1.20 vs 1.2
      - optional spaces around the decimal point: 1 . 2
    """
    s = val_str
    sign = ""
    if s and s[0] in "+-":
        sign, s = s[0], s[1:]

    if "." in s:
        base, frac = s.split(".", 1)
        # allow optional leading 0 for values like 0.22 -> .22 also OK
        if base == "0":
            base_pat = r"(?:0\s*)?"
        else:
            base_pat = re.escape(base)

        # allow spaces around the dot and trailing zeros in fraction
        frac_pat = re.escape(frac) + r"0*"
        num_pat = rf"{base_pat}\s*\.\s*{frac_pat}"
    else:
        # integer: allow .0+ suffix (e.g., 2 or 2.0 or 2.000)
        num_pat = re.escape(s) + r"(?:\s*\.\s*0+)?"

    if sign:
        num_pat = re.escape(sign) + r"\s*" + num_pat

    # final pattern: 'value' with optional : or = and spaces, then tolerant number
    # new: accept "value" or "at"
    return rf"\b(?:value|at)\s*[:=]?\s*{num_pat}\b"

# =========================
# Verifier (numerical fidelity) — exact-string checks
# =========================
def _verify_explanation(text: str, label: int, proba: float, drivers: list) -> (bool, str):
    """
    Accept if:
      - Probability percentage appears (rounded whole number).
      - Each driver name appears.
      - Each driver includes the exact substrings: 'value <value_str>' and 'SHAP <shap_str>'.
    """
    def norm(s): return s.replace("−", "-")
    t = norm(text)

    # 1) Probability (allow '99%' or '99.0%')
    pct_int = int(round(proba * 100))
    prob_patterns = [rf"{pct_int}%\)?", rf"{pct_int}\.0%\)?"]
    if not any(re.search(p, t) for p in prob_patterns):
        return False, f"probability {pct_int}% missing"

    # 2) Per-driver exact-string checks
    for d in drivers:
        name = d["name"]

        # Driver name (case-insensitive, word boundary)
        if not re.search(rf"\b{re.escape(name)}\b", t, re.I):
            return False, f"name missing: {name}"

        val_pat  = _make_value_pattern(d['value_str'])
        shap_pat = rf"(?:\bshap\b\s*[:=]?\s*{re.escape(d['shap_str'])}\b|\(\s*{re.escape(d['shap_str'])}\s*\))"

        if not re.search(val_pat, t, re.I):
            return False, f"value string missing for {d['name']}"
        if not re.search(shap_pat, t, re.I):
            return False, f"SHAP string missing for {d['name']}"

    # 3) Keep the “no comparative language” constraint (optional)
    for w in BANNED_WORDS:
        if re.search(rf"\b{re.escape(w)}\b", t, re.I):
            return False, f"banned word present: {w}"

    return True, "ok"

# =========================
# Public API
# =========================
def explain_row(row: dict):
    """
    row must contain keys in FEATURES, e.g.:
      {
        "yards_per_play_diff": 1.2, "turnovers_diff": -1, "passing_yards": 285,
        "rushing_yards": 115, "sacks": 2, "penalties": 5, "third_down_rate": 0.44
      }
    Returns: (label:int, proba:float, top:list, text:str)
    """
    # Predict
    X = pd.DataFrame([row])[FEATURES]
    proba = float(xgb.predict_proba(X)[:, 1][0])
    label = int(proba >= 0.5)

    # SHAP top-3 by |contribution|
    shap_vals = explainer.shap_values(X)[0]
    idx = np.argsort(-np.abs(shap_vals))[:3]
    top = [{"feature": FEATURES[i], "contribution": float(shap_vals[i])} for i in idx]

    # Enrich with values + direction + friendly names (+ exact strings)
    top_verbose = []
    for t_ in top:
        feat = t_["feature"]
        val = float(X.iloc[0][feat])
        contrib = float(t_["contribution"])
        direction = "positive" if contrib > 0 else ("negative" if contrib < 0 else "neutral")
        top_verbose.append({
            "feature": feat,
            "name": FRIENDLY.get(feat, feat),
            "value": val,
            "value_str": _fmt_val(val),            # <<< added
            "contribution": contrib,
            "shap_str": f"{contrib:+.2f}",         # <<< added
            "direction": direction,
        })

    # Build LLM payload + prompt
    payload = {
        "prediction": label,
        "probability": proba,
        "drivers": [
            {
                "name": d["name"],
                "feature": d["feature"],
                "value": d["value"],
                "value_str": _fmt_val(d["value"]),
                "shap": d["contribution"],
                "shap_str": f"{d['contribution']:+.2f}",
                "direction": d["direction"],
            } for d in top_verbose
        ],
        "phrase_pos": PHRASE_POS,
        "phrase_neg": PHRASE_NEG,
        "connectors": CONNECTORS
    }
    allowed_names = [d["name"] for d in top_verbose]
    messages = _build_messages(payload, allowed_names)
    prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Tokenize & prepare stopping tokens
    inputs = tok(prompt_text, return_tensors="pt", padding=False, add_special_tokens=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    eos_ids = [tok.eos_token_id]
    try:
        eot_id = tok.convert_tokens_to_ids("<|eot_id|>")
        if eot_id is not None and eot_id != tok.eos_token_id:
            eos_ids.append(eot_id)
    except Exception:
        pass

    gen_kwargs = dict(
        max_new_tokens=150,
        do_sample=(XAI_VARIETY != "off"),
        eos_token_id=eos_ids,
        pad_token_id=tok.pad_token_id,
    )
    if XAI_VARIETY != "off":
        gen_kwargs.update(
            temperature=TEMP,
            top_p=TOP_P,
            top_k=TOP_K,
            repetition_penalty=REP_PEN,
        )

    # ===== Auto-retry loop (generate -> verify) =====
    ok = False
    reason = "unknown"
    reply = ""

    for attempt in range(1, max(1, XAI_RETRIES) + 1):
        # per-attempt seeding (so retries explore different samples when sampling is on)
        if XAI_VARIETY in ("stable", "random"):
            import hashlib, os as _os, random as _pyrand
            if XAI_VARIETY == "stable":
                base_seed = int(
                    hashlib.blake2b(
                        json.dumps(payload, sort_keys=True).encode(), digest_size=4
                    ).hexdigest(),
                    16,
                )
                seed = base_seed + (attempt - 1)
            else:
                seed = int.from_bytes(_os.urandom(4), "little")

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            _pyrand.seed(seed)

        with torch.no_grad():
            out_ids = model.generate(**inputs, **gen_kwargs)

        decoded = tok.decode(out_ids[0], skip_special_tokens=True)
        reply = _extract_assistant_reply(decoded, prompt_text)

        ok, reason = _verify_explanation(reply, label, proba, top_verbose)
        if ok:
            break
    # ===== End auto-retry loop =====

    if DEBUG_XAI:
        print("\n[debug] RAW LLM REPLY:\n", reply, "\n")
        print("[debug] verify:", ok, reason)

    if not ok:
        fail = (
            "LLM explanation failed verification: "
            f"{reason}. "
            "Try again or adjust the prompt constraints."
        )
        return label, proba, top_verbose, fail

    return label, proba, top_verbose, reply

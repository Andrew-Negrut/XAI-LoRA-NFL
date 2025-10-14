import json
from pathlib import Path

SRC = Path("shap/shap_samples.jsonl")
OUT = Path("llm/explain_train.jsonl")

def make_output(r):
    label = "Home win" if r["pred_label"]==1 else "Home loss"
    prob  = f"{r['prediction']*100:.0f}%"
    pos = [t for t in r["top_features"] if t["contribution"]>0]
    neg = [t for t in r["top_features"] if t["contribution"]<0]
    lines = [f"Prediction: {label} ({prob})"]
    if pos:
        lines.append("Positive factors:")
        for t in pos: lines.append(f"- {t['feature']} (+{abs(t['contribution']):.2f})")
    if neg:
        lines.append("Negative factors:")
        for t in neg: lines.append(f"- {t['feature']} (-{abs(t['contribution']):.2f})")
    lines.append("Summary: The probability reflects the net effect of these drivers.")
    return "\n".join(lines)

with open(SRC,"r",encoding="utf-8") as fin, open(OUT,"w",encoding="utf-8") as fout:
    for line in fin:
        r = json.loads(line)
        prompt = {
          "prediction": r["pred_label"],
          "probability": r["prediction"],
          "top_features": r["top_features"]
        }
        item = {
          "instruction": "Write a concise, factual explanation of the prediction using the feature attributions.",
          "input": json.dumps(prompt, ensure_ascii=False),
          "output": make_output(r)
        }
        fout.write(json.dumps(item, ensure_ascii=False)+"\n")
print("Wrote", OUT)

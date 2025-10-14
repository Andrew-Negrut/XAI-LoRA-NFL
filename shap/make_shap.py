import pandas as pd, numpy as np, json
from joblib import load
from pathlib import Path
import shap

DATA = Path("data/processed.csv")
MODEL_FILE = Path("models/xgb_homewin.joblib")
OUT = Path("shap/shap_samples.jsonl")

df = pd.read_csv(DATA)
model, FEATURES = load(MODEL_FILE)

X = df[FEATURES]
y = df["home_win"].values

explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X)  # [n_samples, n_features]
proba = model.predict_proba(X)[:,1]
label = (proba >= 0.5).astype(int)

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    for i in range(len(X)):
        vals = sv[i]
        idx = np.argsort(-np.abs(vals))[:3]  # top 3
        top = [{"feature": FEATURES[j], "contribution": float(vals[j])} for j in idx]
        rec = {
          "features": {k: float(X.iloc[i][k]) for k in FEATURES},
          "prediction": float(proba[i]),
          "pred_label": int(label[i]),
          "top_features": top,
          "target": int(y[i])
        }
        f.write(json.dumps(rec)+"\n")
print("Wrote", OUT)

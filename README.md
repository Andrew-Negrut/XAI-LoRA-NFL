# Explainable AI: SHAP-Based NFL Win Explanations (XGBoost + LoRA)

**Live demo:** https://huggingface.co/spaces/anegrut/XAI-LoRA-NFL

**What it does:** Predicts NFL home wins with XGBoost and explains *why* using a LoRA-tuned LLM. Explanations are numerically faithful via a verifier (checks exact “value X” and “SHAP ±Y.YY” strings and auto-retries on failure).

## Quickstart
```bash
conda create -n nfl python=3.10 -y && conda activate nfl
pip install -r requirements.txt
python app/app.py

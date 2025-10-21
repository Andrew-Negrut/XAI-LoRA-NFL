# import os
# import json
# import gradio as gr

# from pipeline import explain_row, FEATURES, FRIENDLY  # FRIENDLY is optional

# def _friendly_label(feat: str) -> str:
#     return FRIENDLY.get(feat, feat)

# # Build Gradio inputs in the exact order of FEATURES
# INPUTS = [gr.Number(label=_friendly_label(f), precision=3) for f in FEATURES]

# def run_explain(*vals):
#     # Map UI values back to a row dict in FEATURES order
#     row = {feat: float(val) for feat, val in zip(FEATURES, vals)}

#     label, proba, top, text = explain_row(row)

#     # Pretty drivers for JSON panel
#     drivers_json = [
#         {
#             "feature": d["feature"],
#             "name": d["name"],
#             "value": d["value"],
#             "shap": round(d["contribution"], 2),
#             "direction": d["direction"],
#         }
#         for d in top
#     ]

#     pred_str = "Home win" if label == 1 else "Home loss"
#     prob_str = f"{proba * 100:.1f}%"

#     return pred_str, prob_str, drivers_json, text

# # Example rows (A = strong win, B = strong loss) mapped in FEATURES order
# example_row_a = {
#     "yards_per_play_diff": 1.2,
#     "turnovers_diff": -1,
#     "passing_yards": 285.0,
#     "rushing_yards": 115.0,
#     "sacks": 2.0,
#     "penalties": 5.0,
#     "third_down_rate": 0.44,
# }
# example_row_b = {
#     "yards_per_play_diff": -2.5,
#     "turnovers_diff": 3.0,
#     "passing_yards": 180.0,
#     "rushing_yards": 75.0,
#     "sacks": 4.0,
#     "penalties": 8.0,
#     "third_down_rate": 0.22,
# }
# EXAMPLES = [
#     [example_row_a[f] for f in FEATURES],
#     [example_row_b[f] for f in FEATURES],
# ]

# CSS = """
# /* Compact spacing so inputs + examples are visible above the fold */
# .gradio-container { gap: 0.5rem !important; }
# section { padding-top: 0.3rem !important; padding-bottom: 0.3rem !important; }
# footer { margin-top: 0.25rem !important; }
# .prose :where(p):not(:where(.not-prose, .not-prose *)) { margin: 0.25rem 0 !important; }

# /* Tighter number inputs */
# input[type="number"] { padding: 0.25rem 0.5rem !important; }

# /* Explanation: high-contrast & prominent */
# #exp_box textarea {
#   background: #0b1220 !important;           /* deep slate */
#   color: #e6ecff !important;                /* light text */
#   border: 2px solid #6366f1 !important;     /* indigo border */
#   box-shadow: 0 0 0 3px rgba(99,102,241,.25) !important;
#   font-weight: 600 !important;
#   font-size: 1.06rem !important;
#   line-height: 1.45 !important;
# }

# /* JSON panel never grows too tall; scroll inside */
# #drivers_json, #drivers_json .wrap, #drivers_json pre, #drivers_json code, #drivers_json .json {
#   max-height: 220px !important;
#   overflow: auto !important;
#   border: 1px solid rgba(148,163,184,.4) !important;
#   border-radius: 8px !important;
# }

# /* Output labels pop */
# #drivers_json label, #exp_box label, #pred_out label, #prob_out label {
#   color: #0f766e !important;   /* teal */
#   font-weight: 800 !important;
# }

# /* Make Explain button compact and keep it with the inputs (not full-width) */
# #explain_btn button {
#   padding: 0.55rem 1rem !important;
#   font-weight: 700 !important;
# }
# """

# with gr.Blocks(
#     css=CSS,
#     title="NFL Outcome Explainer — XGBoost + SHAP + LoRA-Tuned Qwen (Verified XAI)"
# ) as demo:
#     gr.Markdown(
#         """
# # NFL Outcome Explainer — XGBoost + SHAP + LoRA-Tuned Qwen (Verified XAI)
# **Stack:** Tabular ML (XGBoost), explainability (SHAP), parameter-efficient fine-tuning (LoRA on Qwen), prompt + hard numeric verification.

# 1) Click an example (or edit inputs). 2) Press **Explain**.
#         """
#     )

#     # Inputs in two columns, with Explain button anchored under the right column
#     with gr.Row(equal_height=True):
#         mid = (len(INPUTS) + 1) // 2
#         with gr.Column(scale=1):
#             for comp in INPUTS[:mid]:
#                 comp.render()
#         with gr.Column(scale=1):
#             for comp in INPUTS[mid:]:
#                 comp.render()
#             run_btn = gr.Button("Explain", variant="primary", elem_id="explain_btn")

#     # Examples stay visible; do NOT auto-run (user must click Explain)
#     gr.Examples(
#         examples=EXAMPLES,
#         inputs=INPUTS,
#         label="Examples (click to fill inputs, then press Explain)",
#         run_on_click=False,
#         cache_examples=False,
#     )

#     # Outputs side-by-side so JSON never pushes the explanation down
#     with gr.Row(equal_height=True):
#         with gr.Column(scale=1):
#             pred_out = gr.Textbox(label="Prediction", elem_id="pred_out")
#             prob_out = gr.Textbox(label="Win probability", elem_id="prob_out")
#             json_out = gr.JSON(label="Top drivers (name, value, SHAP, direction)", elem_id="drivers_json")
#         with gr.Column(scale=1):
#             exp_out = gr.Textbox(label="Explanation", lines=6, elem_id="exp_box")

#     run_btn.click(
#         run_explain,
#         inputs=INPUTS,
#         outputs=[pred_out, prob_out, json_out, exp_out],
#     )

# if __name__ == "__main__":
#     demo.launch(server_name="127.0.0.1", server_port=7860, share=False)

import os
import json
import gradio as gr

from pipeline import explain_row, FEATURES, FRIENDLY  # FRIENDLY is optional

def _friendly_label(feat: str) -> str:
    return FRIENDLY.get(feat, feat)

# Build Gradio inputs in the exact order of FEATURES
INPUTS = [gr.Number(label=_friendly_label(f), precision=3) for f in FEATURES]

def run_explain(*vals):
    # Map UI values back to a row dict in FEATURES order
    row = {feat: float(val) for feat, val in zip(FEATURES, vals)}

    label, proba, top, text = explain_row(row)

    # Pretty drivers for JSON panel
    drivers_json = [
        {
            "feature": d["feature"],
            "name": d["name"],
            "value": d["value"],
            "shap": round(d["contribution"], 2),
            "direction": d["direction"],
        }
        for d in top
    ]

    pred_str = "Home win" if label == 1 else "Home loss"
    prob_str = f"{proba * 100:.1f}%"

    return pred_str, prob_str, drivers_json, text

# Example rows (A = strong win, B = strong loss) mapped in FEATURES order
example_row_a = {
    "yards_per_play_diff": 1.2,
    "turnovers_diff": -1,
    "passing_yards": 285.0,
    "rushing_yards": 115.0,
    "sacks": 2.0,
    "penalties": 5.0,
    "third_down_rate": 0.44,
}
example_row_b = {
    "yards_per_play_diff": -2.5,
    "turnovers_diff": 3.0,
    "passing_yards": 180.0,
    "rushing_yards": 75.0,
    "sacks": 4.0,
    "penalties": 8.0,
    "third_down_rate": 0.22,
}
EXAMPLES = [
    [example_row_a[f] for f in FEATURES],
    [example_row_b[f] for f in FEATURES],
]

CSS = """
/* Compact spacing so inputs + examples are visible above the fold */
.gradio-container { gap: 0.5rem !important; }
section { padding-top: 0.3rem !important; padding-bottom: 0.3rem !important; }
footer { margin-top: 0.25rem !important; }
.prose :where(p):not(:where(.not-prose, .not-prose *)) { margin: 0.25rem 0 !important; }

/* Tighter number inputs */
input[type="number"] { padding: 0.25rem 0.5rem !important; }

/* Big, high-contrast CTA that gently pulses */
#cta {
  background: #fde047 !important;             /* amber-300 */
  color: #111827 !important;                  /* slate-900 */
  border: 3px solid #f59e0b !important;       /* amber-500 */
  border-radius: 14px !important;
  padding: 0.9rem 1.1rem !important;
  font-weight: 900 !important;
  font-size: 1.15rem !important;
  letter-spacing: .2px !important;
  box-shadow: 0 6px 22px rgba(245, 158, 11, .35) !important;
  animation: ctaPulse 2.4s ease-in-out infinite;
}
#cta strong { text-decoration: underline; }
@keyframes ctaPulse {
  0%, 100% { transform: scale(1); }
  50%      { transform: scale(1.025); }
}
@media (prefers-reduced-motion: reduce) {
  #cta { animation: none; }
}

/* Explanation: high-contrast & prominent (unchanged) */
#exp_box textarea {
  background: #0b1220 !important;           /* deep slate */
  color: #e6ecff !important;                /* light text */
  border: 2px solid #6366f1 !important;     /* indigo border */
  box-shadow: 0 0 0 3px rgba(99,102,241,.25) !important;
  font-weight: 600 !important;
  font-size: 1.06rem !important;
  line-height: 1.45 !important;
}

/* JSON panel never grows too tall; scroll inside (unchanged) */
#drivers_json, #drivers_json .wrap, #drivers_json pre, #drivers_json code, #drivers_json .json {
  max-height: 220px !important;
  overflow: auto !important;
  border: 1px solid rgba(148,163,184,.4) !important;
  border-radius: 8px !important;
}

/* Output labels pop (unchanged) */
#drivers_json label, #exp_box label, #pred_out label, #prob_out label {
  color: #0f766e !important;   /* teal */
  font-weight: 800 !important;
}

/* Make Explain button compact and keep it with the inputs (unchanged) */
#explain_btn button {
  padding: 0.55rem 1rem !important;
  font-weight: 700 !important;
}

/* >>> Make the Examples header HUGE and eye-catching <<< */
#examples_panel .label, #examples_panel label {
  font-size: 1.15rem !important;
  font-weight: 900 !important;
  color: #f97316 !important;                 /* orange-500 */
  text-transform: uppercase !important;
  letter-spacing: .6px !important;
  text-shadow: 0 2px 10px rgba(249, 115, 22, .35) !important;
  display: inline-block !important;
  padding: 0.25rem 0.55rem !important;
  border-left: 6px solid #fb923c !important; /* orange-400 */
  background: rgba(251, 146, 60, .1) !important;
  border-radius: 6px !important;
}

/* Slight glow on the examples table so your eye falls there */
#examples_panel .table-wrap, #examples_panel table {
  box-shadow: 0 0 0 3px rgba(251, 146, 60, .15) inset !important;
  border-radius: 8px !important;
}
"""

with gr.Blocks(
    css=CSS,
    title="NFL Outcome Explainer — XGBoost + SHAP + LoRA-Tuned Qwen (Verified XAI)"
) as demo:
    # High-contrast animated CTA
    gr.Markdown(
        "👉 **Start here:** click one of the example rows below to prefill inputs, then press **Explain**.",
        elem_id="cta"
    )

    # Inputs in two columns, with Explain button anchored under the right column
    with gr.Row(equal_height=True):
        mid = (len(INPUTS) + 1) // 2
        with gr.Column(scale=1):
            for comp in INPUTS[:mid]:
                comp.render()
        with gr.Column(scale=1):
            for comp in INPUTS[mid:]:
                comp.render()
            run_btn = gr.Button("Explain", variant="primary", elem_id="explain_btn")

    # Examples stay visible; do NOT auto-run (user must click Explain)
    gr.Examples(
        examples=EXAMPLES,
        inputs=INPUTS,
        label="Examples (click to fill inputs, then press Explain)",
        run_on_click=False,
        cache_examples=False,
        elem_id="examples_panel"   # <<< so we can style the label to be big + bright
    )

    # Outputs side-by-side so JSON never pushes the explanation down
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            pred_out = gr.Textbox(label="Prediction", elem_id="pred_out")
            prob_out = gr.Textbox(label="Win probability", elem_id="prob_out")
            json_out = gr.JSON(label="Top drivers (name, value, SHAP, direction)", elem_id="drivers_json")
        with gr.Column(scale=1):
            exp_out = gr.Textbox(label="Explanation", lines=6, elem_id="exp_box")

    run_btn.click(
        run_explain,
        inputs=INPUTS,
        outputs=[pred_out, prob_out, json_out, exp_out],
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)

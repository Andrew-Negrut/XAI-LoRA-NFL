from pipeline import explain_row

row = {
    "yards_per_play_diff": -2.5,
    "turnovers_diff": 3.0,
    "passing_yards": 150.0,
    "rushing_yards": 60.0,
    "sacks": 6.0,
    "penalties": 11.0,
    "third_down_rate": 0.22,
}

label, proba, top, text = explain_row(row)
print("label:", label)
print("proba:", proba)
print("top:", top)
print("---")
print(text)

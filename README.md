# NFL Play Predictor

A machine learning model that predicts whether an NFL offense will pass or rush 
on a given play, trained on 2024 NFL play-by-play data. Built in Python/Google Colab.

## Motivation
Play-calling prediction is a core problem in football analytics — if a defense can 
anticipate a pass or run before the snap, it changes everything. I built this to see 
how well a Random Forest model could predict play type using only pre-snap situational 
data available to both teams.

## Methodology
- **Model:** Random Forest Classifier (scikit-learn, 80/20 train/test split)
- **Platform:** Python in Google Colab
- **Data Source:** NFLSavant 2024 play-by-play dataset
- **Features:** Offensive team, down, yards to go, quarter, minute, yard line
- **Target:** IsPass (binary — pass or rush)
- **Engineering:** Filtered out non-pass/non-rush plays, removed scrambles, 
  one-hot encoding for categorical features, sklearn Pipeline for clean inference

## How to Run
Open directly in Google Colab:
[Open in Google Colab](https://colab.research.google.com/drive/109Y-lwjZE1TT_oNB2cBTZ8P4ZshoHWrr)

Or clone and run locally:
```bash
pip install pandas scikit-learn
python nfl_data_engineering.py
```

## Example Output
Given pre-snap situation for MIN (3rd down, varying yards to go):

| Down | ToGo | Quarter | Minute | YardLine | P(Pass) |
|------|------|---------|--------|----------|---------|
| 3 | 10 | 2 | 12 | 40 | ~high |
| 3 | 1 | 2 | 12 | 40 | ~low |
| 3 | 4 | 2 | 12 | 40 | ~mid |

*(Run the model to see exact probabilities)*

## What I'd Improve Next
- Add defensive formation and personnel grouping as features
- Expand to multi-class prediction (pass, rush, scramble, screen, etc.)
- Train across multiple seasons to reduce year-specific variance
- Add SHAP values for per-prediction feature explanation

Intern Performance Prediction Model

Objective
Build a machine learning model to predict intern performance based on task
completion rates and feedback, in order to identify which interns are likely
to excel or struggle.

Dataset
A sample dataset of 20 interns with the following features:
- `completion_time_days` — how long the intern took to complete tasks (lower = faster)
- `feedback_rating` — mentor feedback score (out of 5)
- `attendance_pct` — attendance percentage
- `performance_score` — target variable (0–100), what the model predicts

Approach
1. **Data loading & exploration** — loaded the dataset with pandas, checked for
   missing values and reviewed summary statistics.
2. **Train/test split** — 80/20 split, so the model is evaluated on data it
   never saw during training.
3. **Models trained** — Random Forest Regressor and Gradient Boosting Regressor
   (XGBoost-equivalent), to compare two ensemble approaches.
4. **Evaluation** — MAE, RMSE, and R² score, since this is a regression problem
   (predicting a continuous score, not a category).
5. **Feature importance** — checked which factor (completion time, feedback,
   or attendance) influenced predictions most.
6. **Testing on new data** — ran the trained model on 3 made-up interns it had
   never seen, to confirm its predictions made intuitive sense.

## Results
| Model | MAE | RMSE | R² |
|---|---|---|---|
| Random Forest | 11.51 | 11.86 | 0.47 |
| Gradient Boosting | 9.96 | 10.12 | 0.61 |

**Gradient Boosting performed better** on this dataset, with lower error and
a higher R² score.

**Feature importance (Random Forest):**
- Feedback rating: 41%
- Completion time: 32%
- Attendance: 27%

Feedback rating was the strongest predictor of performance.

## Test on new, unseen interns
| Intern | Completion Time | Feedback | Attendance | Predicted Score |
|---|---|---|---|---|
| Ali (fast, great feedback) | 3 days | 4.8 | 95% | 93.2 |
| Zara (slow, poor attendance) | 13 days | 2.2 | 65% | 57.0 |
| Hamza (average all-round) | 8 days | 3.5 | 80% | 74.0 |

The predictions align with intuition — strong performers score high, weaker
performers score low — which suggests the model learned meaningful patterns
despite the small dataset size.

## Note on dataset size
This project uses a small (20-row) sample dataset. With more real-world data,
model performance (especially R²) would likely improve and become more
reliable. This is flagged as a natural next step rather than a limitation
hidden from the results.

## Files
- `intern_data.csv` — the dataset
- `step1_load_split.py` — load data, explore it, train/test split
- `step2_train_evaluate.py` — train both models, evaluate, feature importance
- `step3_visualize.py` — predicted vs actual scatter plot
- `step4_test_new_data.py` — test the trained model on new, unseen interns
- `predicted_vs_actual.png` — visualization output

## How to run
```
pip install pandas scikit-learn xgboost matplotlib
python3 step1_load_split.py
python3 step2_train_evaluate.py
python3 step3_visualize.py
python3 step4_test_new_data.py
```

Note: `step2_train_evaluate.py` uses `GradientBoostingRegressor` from
scikit-learn as an XGBoost equivalent. To use real XGBoost, swap in:
```python
from xgboost import XGBRegressor
gb_model = XGBRegressor(n_estimators=100, random_state=42)
```

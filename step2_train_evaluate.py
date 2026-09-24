import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Same setup as Step 1 ---
df = pd.read_csv('intern_data.csv')
X = df[['completion_time_days', 'feedback_rating', 'attendance_pct']]
y = df['performance_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- MODEL 1: RANDOM FOREST ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)          # learns patterns from training data
rf_predictions = rf_model.predict(X_test)  # predicts on unseen test data

# --- MODEL 2: GRADIENT BOOSTING (same family as XGBoost) ---
# NOTE: On your own machine, run: pip install xgboost
# then: from xgboost import XGBRegressor
# and:  xgb_model = XGBRegressor(n_estimators=100, random_state=42)
# It's used exactly the same way as below.
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
gb_predictions = gb_model.predict(X_test)

# --- EVALUATE BOTH MODELS ---
def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    print(f"\n{name}")
    print(f"  MAE  (avg error in points): {mae:.2f}")
    print(f"  RMSE (penalizes big misses more): {rmse:.2f}")
    print(f"  R^2  (0-1, higher = better fit): {r2:.2f}")

print("Actual test values: ", y_test.values)
print("RF predictions:     ", rf_predictions.round(1))
print("GB predictions:     ", gb_predictions.round(1))

evaluate("Random Forest", y_test, rf_predictions)
evaluate("Gradient Boosting", y_test, gb_predictions)

# --- FEATURE IMPORTANCE (which factor mattered most) ---
print("\nFeature importance (Random Forest):")
for feature, importance in zip(X.columns, rf_model.feature_importances_):
    print(f"  {feature}: {importance:.2f}")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

# --- Train the model (same as before) ---
df = pd.read_csv('intern_data.csv')
X = df[['completion_time_days', 'feedback_rating', 'attendance_pct']]
y = df['performance_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ============================================================
# TEST ON BRAND-NEW, MADE-UP INTERNS (not in the training data)
# This is what "testing the model" looks like in practice.
# ============================================================
new_interns = pd.DataFrame({
    'intern_name': ['Ali (fast, great feedback)', 'Zara (slow, poor attendance)', 'Hamza (average all-round)'],
    'completion_time_days': [3, 13, 8],
    'feedback_rating': [4.8, 2.2, 3.5],
    'attendance_pct': [95, 65, 80]
})

predictions = model.predict(new_interns[['completion_time_days', 'feedback_rating', 'attendance_pct']])

new_interns['predicted_performance_score'] = predictions.round(1)

print("Testing the trained model on 3 brand-new interns:\n")
print(new_interns.to_string(index=False))

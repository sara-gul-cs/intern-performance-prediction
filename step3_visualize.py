import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

df = pd.read_csv('intern_data.csv')
X = df[['completion_time_days', 'feedback_rating', 'attendance_pct']]
y = df['performance_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Scatter plot: perfect predictions would fall exactly on the diagonal line
plt.figure(figsize=(6, 6))
plt.scatter(y_test, predictions, color='steelblue', s=100, label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', label='Perfect prediction line')
plt.xlabel('Actual Performance Score')
plt.ylabel('Predicted Performance Score')
plt.title('Predicted vs Actual Intern Performance')
plt.legend()
plt.tight_layout()
plt.savefig('predicted_vs_actual.png', dpi=150)
print("Saved plot as predicted_vs_actual.png")

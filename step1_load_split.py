import pandas as pd
from sklearn.model_selection import train_test_split

# 1. LOAD THE DATA
df = pd.read_csv('intern_data.csv')
print("First 5 rows:")
print(df.head())

print("\nBasic info:")
print(df.info())

print("\nSummary statistics:")
print(df.describe())

# 2. SEPARATE FEATURES (X) FROM TARGET (y)
# Features = what we use to predict. Target = what we want to predict.
X = df[['completion_time_days', 'feedback_rating', 'attendance_pct']]
y = df['performance_score']

# 3. TRAIN/TEST SPLIT
# We train on 80% of data, and test on the 20% the model has NEVER seen.
# random_state=42 just makes the split reproducible (same split every time you run it).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")

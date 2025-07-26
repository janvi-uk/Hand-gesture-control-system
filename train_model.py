import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# Updated file path
data_file_path = 'final_product.csv'
model_save_path = 'gesture_model.pkl'

# Check if the CSV file exists
if not os.path.exists(data_file_path):
    print(f"Error: Data file not found at {data_file_path}.")
    exit()

# Load the CSV data
try:
    df = pd.read_csv(data_file_path)
except Exception as e:
    print(f"Error loading CSV file: {e}")
    exit()

# Ensure 'label' column is present
if "label" not in df.columns:
    print("Error: 'label' column not found in dataset.")
    exit()

X = df.drop("label", axis=1)
y = df["label"]

# Data validation
if len(X) == 0:
    print("Error: Dataset is empty.")
    exit()

if len(y.unique()) < 2:
    print("Error: At least two unique labels are required. Found:", y.unique())
    exit()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
print("\nTraining RandomForestClassifier...")
model.fit(X_train, y_train)
print("Training complete.")

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(report)

# Save model
with open(model_save_path, 'wb') as f:
    pickle.dump(model, f)

print(f"\nModel saved successfully to {model_save_path}")




import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load the data
print("Loading dataset...")
df = pd.read_excel('aryan.xlsx')
print(f"Dataset shape: {df.shape}")

# Check if the file is empty (only headers, no data)
if df.shape[0] == 0:
    print("Error: The Excel file is empty. Please add data to the file before training.")
    exit()

# Convert all columns to numeric, errors='coerce' will convert non-numeric values to NaN
print("\nConverting columns to numeric...")
for column in df.columns:
    df[column] = pd.to_numeric(df[column], errors='coerce')

# Show number of missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Fill NaN values with mean of respective columns
print("\nFilling NaN values with mean...")
df = df.fillna(df.mean())

# Split features and target
X = df.drop('Potability', axis=1)
y = df['Potability']

# Split the data into training and testing sets
print("\nSplitting into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Scale the features
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
print("\nTraining Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=42
)
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf_model.predict(X_test_scaled)

# Print model performance
print("\nModel Performance:")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model and scaler
print("\nSaving model and scaler...")
joblib.dump(rf_model, 'water_quality_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

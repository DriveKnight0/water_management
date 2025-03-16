import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

print("Loading enhanced dataset...")
df = pd.read_excel('water_quality_combined.xlsx')
print(f"Dataset shape: {df.shape}")

# Convert all columns to numeric
print("\nConverting columns to numeric...")
for column in df.columns:
    df[column] = pd.to_numeric(df[column], errors='coerce')

# Fill NaN values with mean
print("\nFilling NaN values with mean...")
df = df.fillna(df.mean())

# Split features and target with 70-30 ratio
print("\nSplitting into train and test sets (70-30)...")
X = df.drop('Potability', axis=1)
y = df['Potability']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Scale the features
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest with enhanced parameters
print("\nTraining Enhanced Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=300,  # Increased number of trees
    max_depth=10,      # Slightly increased depth
    min_samples_split=8,
    min_samples_leaf=4,
    max_features='sqrt',
    class_weight='balanced',  # Added class weights
    random_state=42
)

# Perform cross-validation
print("\nPerforming cross-validation...")
cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Average CV score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

# Train the final model
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf_model.predict(X_test_scaled)

# Print model performance
print("\nEnhanced Model Performance:")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model and scaler
print("\nSaving enhanced model and scaler...")
joblib.dump(rf_model, 'water_quality_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Feature importance with percentages
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_ * 100  # Convert to percentage
}).sort_values('importance', ascending=False)

print("\nFeature Importance (%):")
print(feature_importance.to_string(float_format=lambda x: '{:.2f}%'.format(x)))

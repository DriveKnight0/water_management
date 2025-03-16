import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load and preprocess data
def preprocess_data():
    # Read the dataset
    data = pd.read_csv('water_quality_sample.csv')
    
    # Handle missing values
    data = data.replace('ph', np.nan)  # Replace 'ph' strings with NaN
    data = data.dropna()  # Remove rows with missing values
    
    # Add some random noise to make the model more robust
    np.random.seed(42)
    noise_columns = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
    for col in noise_columns:
        noise = np.random.normal(0, data[col].std() * 0.1, size=len(data))
        data[col] = data[col] + noise
    
    # Split features and target
    X = data.drop('Potability', axis=1)
    y = data['Potability']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    joblib.dump(scaler, 'scaler.pkl')
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# Train the model
def train_model(X_train, y_train):
    # Initialize model with parameters focused on generalization
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        random_state=42
    )
    
    # Perform cross-validation
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Average CV score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
    
    # Train the model
    rf_model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(rf_model, 'water_quality_model.pkl')
    
    return rf_model

# Main execution
if __name__ == "__main__":
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data()
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Print detailed performance metrics
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    print("\nTraining Set Performance:")
    print(classification_report(y_train, train_pred))
    
    print("\nTest Set Performance:")
    print(classification_report(y_test, test_pred))
    
    # Print feature importance
    feature_importance = pd.DataFrame({
        'feature': ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'],
        'importance': model.feature_importances_
    })
    print("\nFeature Importance:")
    print(feature_importance.sort_values('importance', ascending=False))

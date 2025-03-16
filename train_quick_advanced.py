import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import KNNImputer
import joblib
import warnings
warnings.filterwarnings('ignore')

def create_interaction_features(df):
    """Create interaction features between important parameters"""
    df['ph_hardness'] = df['ph'] * df['Hardness']
    df['sulfate_conductivity'] = df['Sulfate'] * df['Conductivity']
    return df

def handle_outliers(df, columns, n_sigmas=3):
    """Handle outliers using sigma clipping"""
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = df[col].clip(mean - n_sigmas * std, mean + n_sigmas * std)
    return df

print("Loading dataset...")
df = pd.read_excel('water_quality_combined.xlsx')
print(f"Initial dataset shape: {df.shape}")

# Advanced preprocessing
print("\nPerforming advanced preprocessing...")
for column in df.columns:
    df[column] = pd.to_numeric(df[column], errors='coerce')

# Handle missing values with KNN imputation
imputer = KNNImputer(n_neighbors=5)
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Handle outliers
df = handle_outliers(df, df.columns[:-1])

# Feature engineering
print("\nPerforming feature engineering...")
df = create_interaction_features(df)

# Split features and target
X = df.drop('Potability', axis=1)
y = df['Potability']

# Split the data
print("\nSplitting into train and test sets (70-30)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Advanced feature scaling
print("\nApplying advanced feature scaling...")
scaler = PowerTransformer(method='yeo-johnson')
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create optimized base models
rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=12,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

gb = GradientBoostingClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    min_samples_split=4,
    min_samples_leaf=2,
    subsample=0.9,
    random_state=42
)

# Create and train voting classifier
print("\nTraining ensemble model...")
voting_clf = VotingClassifier(
    estimators=[
        ('rf', rf),
        ('gb', gb)
    ],
    voting='soft'
)

voting_clf.fit(X_train_scaled, y_train)

# Make predictions
y_pred = voting_clf.predict(X_test_scaled)

# Print model performance
print("\nEnhanced Model Performance:")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model and scaler
print("\nSaving enhanced model and scaler...")
joblib.dump(voting_clf, 'water_quality_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Feature importance from Random Forest
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': voting_clf.named_estimators_['rf'].feature_importances_ * 100
}).sort_values('importance', ascending=False)

print("\nFeature Importance (%):")
print(feature_importance.to_string(float_format=lambda x: '{:.2f}%'.format(x)))

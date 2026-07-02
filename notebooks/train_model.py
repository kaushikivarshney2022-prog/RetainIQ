"""
RetainIQ - Customer Churn Prediction - Model Training Script
End-to-End MLOps Pipeline
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_data():
    """
    Load the customer churn dataset
    """
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "dataset", "customer_churn.csv")
    
    if os.path.exists(data_path):
        print(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        # Convert TotalCharges to numeric and handle blank values
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = df['TotalCharges'].astype(str).str.strip()
            df['TotalCharges'].replace('', np.nan, inplace=True)
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            if df['TotalCharges'].isna().any():
                fill_count = int(df['TotalCharges'].isna().sum())
                print(f"  Converting TotalCharges to numeric and filling {fill_count} missing values with 0")
                df['TotalCharges'].fillna(0.0, inplace=True)
        # Normalize target values
        # Normalize target values
if 'Churn' in df.columns:
    df['Churn'] = df['Churn'].astype(str).str.strip()

    # Convert Yes/No to 1/0
    df['Churn'] = df['Churn'].replace({
        'Yes': 1,
        'No': 0,
        'yes': 1,
        'no': 0,
        '1': 1,
        '0': 0
    })

    # Convert to numeric safely
    df['Churn'] = pd.to_numeric(df['Churn'], errors='coerce')

    # Fill missing values
    df['Churn'] = df['Churn'].fillna(0)

    # Convert to integer
    df['Churn'] = df['Churn'].astype(int)
    else:
        print("Creating sample dataset...")
        # Create sample dataset for demonstration
        np.random.seed(42)
        n_samples = 7043
        
        df = pd.DataFrame({
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            'Partner': np.random.choice(['Yes', 'No'], n_samples, p=[0.45, 0.55]),
            'Dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.30, 0.70]),
            'tenure': np.random.randint(1, 72, n_samples),
            'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.90, 0.10]),
            'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples, p=[0.40, 0.45, 0.15]),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.40, 0.45, 0.15]),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.30, 0.45, 0.25]),
            'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.35, 0.40, 0.25]),
            'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.30, 0.45, 0.25]),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.25, 0.50, 0.25]),
            'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.35, 0.40, 0.25]),
            'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.35, 0.40, 0.25]),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.25, 0.20]),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples, p=[0.60, 0.40]),
            'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n_samples),
            'MonthlyCharges': np.random.uniform(20, 120, n_samples).round(2),
            'TotalCharges': np.random.uniform(50, 8000, n_samples).round(2),
        })
        
        # Create churn based on some patterns
        df['Churn'] = 0
        # Higher churn for month-to-month contracts
        month_to_month_mask = df['Contract'] == 'Month-to-month'
        df.loc[month_to_month_mask, 'Churn'] = np.random.choice([0, 1], 
            size=len(df[month_to_month_mask]), p=[0.6, 0.4])
        # Higher churn for fiber optic
        fiber_mask = df['InternetService'] == 'Fiber optic'
        df.loc[fiber_mask, 'Churn'] = np.random.choice([0, 1], 
            size=len(df[fiber_mask]), p=[0.7, 0.3])
        # Lower churn for long tenure
        tenure_mask = df['tenure'] > 50
        df.loc[tenure_mask, 'Churn'] = np.random.choice([0, 1], 
            size=len(df[tenure_mask]), p=[0.9, 0.1])
        
        # Save the dataset
        os.makedirs('dataset', exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"Sample dataset saved to: {data_path}")
    
    return df

def encode_categorical_features(df):
    """
    Encode categorical features matching the Flask app encoding
    """
    print("Encoding categorical features...")
    data = df.copy()
    
    # Gender: Male=0, Female=1
    data['gender_encoded'] = data['gender'].apply(lambda x: 1 if x == 'Female' else 0)
    
    # Senior Citizen: 0 or 1 (already encoded)
    # Keep as is
    
    # Partner: No=0, Yes=1
    data['Partner_encoded'] = data['Partner'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Dependents: No=0, Yes=1
    data['Dependents_encoded'] = data['Dependents'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Phone Service: No=0, Yes=1
    data['PhoneService_encoded'] = data['PhoneService'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Multiple Lines: No=0, Yes=1, No phone service=2
    data['MultipleLines_encoded'] = data['MultipleLines'].apply(
        lambda x: 2 if x == 'No phone service' else (1 if x == 'Yes' else 0)
    )
    
    # Internet Service: DSL=0, Fiber optic=1, No=2
    internet_mapping = {'DSL': 0, 'Fiber optic': 1, 'No': 2}
    data['InternetService_encoded'] = data['InternetService'].map(internet_mapping)
    
    # Online Security: No=0, Yes=1, No internet service=2
    def encode_online_security(x):
        if x == 'No internet service':
            return 2
        return 1 if x == 'Yes' else 0
    data['OnlineSecurity_encoded'] = data['OnlineSecurity'].apply(encode_online_security)
    
    # Online Backup: No=0, Yes=1, No internet service=2
    def encode_online_backup(x):
        if x == 'No internet service':
            return 2
        return 1 if x == 'Yes' else 0
    data['OnlineBackup_encoded'] = data['OnlineBackup'].apply(encode_online_backup)
    
    # Device Protection: No=0, Yes=1, No internet service=2
    def encode_device_protection(x):
        if x == 'No internet service':
            return 2
        return 1 if x == 'Yes' else 0
    data['DeviceProtection_encoded'] = data['DeviceProtection'].apply(encode_device_protection)
    
    # Tech Support: No=0, Yes=1, No internet service=2
    def encode_tech_support(x):
        if x == 'No internet service':
            return 2
        return 1 if x == 'Yes' else 0
    data['TechSupport_encoded'] = data['TechSupport'].apply(encode_tech_support)
    
    # Streaming TV: No=0, Yes=1, No internet service=2
    def encode_streaming_tv(x):
        if x == 'No internet service':
            return 2
        return 1 if x == 'Yes' else 0
    data['StreamingTV_encoded'] = data['StreamingTV'].apply(encode_streaming_tv)
    
    # Streaming Movies: No=0, Yes=1, No internet service=2
    def encode_streaming_movies(x):
        if x == 'No internet service':
            return 2
        return 1 if x == 'Yes' else 0
    data['StreamingMovies_encoded'] = data['StreamingMovies'].apply(encode_streaming_movies)
    
    # Contract: Month-to-month=0, One year=1, Two year=2
    contract_mapping = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    data['Contract_encoded'] = data['Contract'].map(contract_mapping)
    
    # Paperless Billing: No=0, Yes=1
    data['PaperlessBilling_encoded'] = data['PaperlessBilling'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Payment Method: Electronic check=0, Mailed check=1, Bank transfer=2, Credit card=3
    payment_mapping = {
        'Electronic check': 0,
        'Mailed check': 1,
        'Bank transfer (automatic)': 2,
        'Credit card (automatic)': 3
    }
    data['PaymentMethod_encoded'] = data['PaymentMethod'].map(payment_mapping)
    
    # Define feature columns (must match Flask app)
    feature_cols = [
        'gender_encoded', 'SeniorCitizen', 'Partner_encoded', 'Dependents_encoded',
        'tenure', 'PhoneService_encoded', 'MultipleLines_encoded',
        'InternetService_encoded', 'OnlineSecurity_encoded', 'OnlineBackup_encoded',
        'DeviceProtection_encoded', 'TechSupport_encoded', 'StreamingTV_encoded',
        'StreamingMovies_encoded', 'Contract_encoded', 'PaperlessBilling_encoded',
        'PaymentMethod_encoded', 'MonthlyCharges', 'TotalCharges'
    ]
    
    # Feature names for reference
    feature_names = feature_cols
    
    print(f"Encoded {len(feature_cols)} features")
    return data, feature_names

def prepare_data(df):
    """
    Prepare data for training
    """
    print("Preparing data for training...")
    
    # Encode categorical features
    encoded_df, feature_names = encode_categorical_features(df)
    
    # Define features and target
    X = encoded_df[feature_names]
    y = encoded_df['Churn'].map({'No': 0, 'Yes': 1}).astype(int)
    
    # Handle missing values
    X = X.fillna(0)
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Churn distribution: {y.value_counts().to_dict()}")
    
    return X, y, feature_names

def train_models(X_train, X_test, y_train, y_test):
    """
    Train multiple models and select the best one
    """
    print("\nTraining models...")
    
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, 
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=42
        ),
        'SVM': SVC(
            probability=True,
            kernel='rbf',
            C=1.0,
            gamma='scale',
            random_state=42
        )
    }
    
    results = {}
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        print(f"\n  Training {name}...")
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            cv_mean = cv_scores.mean()
            
            results[name] = {
                'model': model,
                'accuracy': score,
                'cv_mean': cv_mean,
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            print(f"    Accuracy: {score:.4f}")
            print(f"    CV Score: {cv_mean:.4f}")
            
            if score > best_score:
                best_score = score
                best_model = model
                
        except Exception as e:
            print(f"    Error: {e}")
            continue
    
    print(f"\nBest model: {best_model.__class__.__name__} with accuracy: {best_score:.4f}")
    
    return best_model, results

def save_artifacts(model, scaler, feature_names):
    """
    Save the trained model and artifacts
    """
    print("\nSaving artifacts...")
    
    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('artifacts', exist_ok=True)
    
    # Save model
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names
    }
    model_path = os.path.join('models', 'model.pkl')
    joblib.dump(model_data, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save artifacts
    artifacts = {
        'scaler': scaler,
        'feature_names': feature_names,
        'model_type': model.__class__.__name__
    }
    artifacts_path = os.path.join('artifacts', 'features.pkl')
    joblib.dump(artifacts, artifacts_path)
    print(f"Artifacts saved to: {artifacts_path}")
    
    return True

def main():
    """
    Main training pipeline
    """
    print("="*60)
    print("RetainIQ - Customer Churn Prediction - Model Training")
    print("="*60)
    
    try:
        # Load data
        print("\nLoading data...")
        df = load_data()
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Prepare data
        X, y, feature_names = prepare_data(df)
        
        # Split data
        print("\nSplitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Train: {len(X_train)} samples")
        print(f"Test: {len(X_test)} samples")
        
        # Scale features
        print("\nScaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print("Features scaled")
        
        # Train models
        best_model, results = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # Save artifacts
        save_artifacts(best_model, scaler, feature_names)
        
        # Print summary
        print("\n" + "="*60)
        print("Training Summary")
        print("="*60)
        for name, result in results.items():
            print(f"\n{name}:")
            print(f"  Accuracy: {result['accuracy']:.4f}")
            print(f"  CV Mean: {result.get('cv_mean', 'N/A'):.4f}")
            if 'classification_report' in result:
                print(f"  Precision: {result['classification_report']['weighted avg']['precision']:.4f}")
                print(f"  Recall: {result['classification_report']['weighted avg']['recall']:.4f}")
                print(f"  F1-Score: {result['classification_report']['weighted avg']['f1-score']:.4f}")
        
        print("\n" + "="*60)
        print("Training completed successfully")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\nError in training pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
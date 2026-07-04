"""
assess.py - Assessment Utility Module
This module provides churn assessment functions for testing and debugging.
Note: Main assessment is handled in app.py
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

class ChurnPredictor:
    """
    Customer Churn Assessment Utility Class
    Used for testing and debugging purposes
    """
    
    def __init__(self, model_path=None, artifacts_path=None):
        """
        Initialize the predictor with model and artifacts
        
        Args:
            model_path: Path to model.pkl file
            artifacts_path: Path to features.pkl file
        """
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        # Set default paths
        if model_path is None:
            model_path = os.path.join('models', 'model.pkl')
        if artifacts_path is None:
            artifacts_path = os.path.join('artifacts', 'features.pkl')
        
        # Load model and artifacts
        self.load_model(model_path, artifacts_path)
    
    def load_model(self, model_path, artifacts_path):
        """
        Load the trained assessment engine and artifacts
        """
        try:
            # Load model
            if os.path.exists(model_path):
                model_data = joblib.load(model_path)
                if isinstance(model_data, dict):
                    self.model = model_data.get('model')
                    self.scaler = model_data.get('scaler')
                    self.feature_names = model_data.get('feature_names')
                else:
                    self.model = model_data
                print(f"✅ Assessment engine loaded from {model_path}")
            else:
                print(f"❌ Assessment engine not found at {model_path}")
                return False
            
            # Load artifacts if not already loaded
            if self.scaler is None and os.path.exists(artifacts_path):
                artifacts = joblib.load(artifacts_path)
                self.scaler = artifacts.get('scaler')
                if self.feature_names is None:
                    self.feature_names = artifacts.get('feature_names')
                print(f"✅ Artifacts loaded from {artifacts_path}")
            
            # Set default feature names if not available
            if self.feature_names is None:
                self.feature_names = [
                    'gender_encoded', 'SeniorCitizen', 'Partner_encoded', 
                    'Dependents_encoded', 'tenure', 'PhoneService_encoded',
                    'MultipleLines_encoded', 'InternetService_encoded',
                    'OnlineSecurity_encoded', 'OnlineBackup_encoded',
                    'DeviceProtection_encoded', 'TechSupport_encoded',
                    'StreamingTV_encoded', 'StreamingMovies_encoded',
                    'Contract_encoded', 'PaperlessBilling_encoded',
                    'PaymentMethod_encoded', 'MonthlyCharges', 'TotalCharges'
                ]
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading assessment engine: {e}")
            return False
    
    def encode_features(self, data):
        """
        Encode categorical features for assessment
        
        Args:
            data: Dictionary containing customer features
        
        Returns:
            numpy array of encoded features
        """
        try:
            # Gender: Male=0, Female=1
            gender_encoded = 1 if data.get('gender') == 'Female' else 0
            
            # Senior Citizen: 0 or 1
            senior_encoded = int(data.get('senior_citizen', 0))
            
            # Partner: No=0, Yes=1
            partner_encoded = 1 if data.get('partner') == 'Yes' else 0
            
            # Dependents: No=0, Yes=1
            dependents_encoded = 1 if data.get('dependents') == 'Yes' else 0
            
            # Phone Service: No=0, Yes=1
            phone_service_encoded = 1 if data.get('phone_service') == 'Yes' else 0
            
            # Multiple Lines: No=0, Yes=1, No phone service=2
            multiple_lines = data.get('multiple_lines', 'No')
            if multiple_lines == 'No phone service':
                multiple_lines_encoded = 2
            else:
                multiple_lines_encoded = 1 if multiple_lines == 'Yes' else 0
            
            # Internet Service: DSL=0, Fiber optic=1, No=2
            internet_service = data.get('internet_service', 'DSL')
            internet_service_encoded = {'DSL': 0, 'Fiber optic': 1, 'No': 2}.get(internet_service, 0)
            
            # Online Security: No=0, Yes=1, No internet service=2
            online_security = data.get('online_security', 'No')
            if online_security == 'No internet service':
                online_security_encoded = 2
            else:
                online_security_encoded = 1 if online_security == 'Yes' else 0
            
            # Online Backup: No=0, Yes=1, No internet service=2
            online_backup = data.get('online_backup', 'No')
            if online_backup == 'No internet service':
                online_backup_encoded = 2
            else:
                online_backup_encoded = 1 if online_backup == 'Yes' else 0
            
            # Device Protection: No=0, Yes=1, No internet service=2
            device_protection = data.get('device_protection', 'No')
            if device_protection == 'No internet service':
                device_protection_encoded = 2
            else:
                device_protection_encoded = 1 if device_protection == 'Yes' else 0
            
            # Tech Support: No=0, Yes=1, No internet service=2
            tech_support = data.get('tech_support', 'No')
            if tech_support == 'No internet service':
                tech_support_encoded = 2
            else:
                tech_support_encoded = 1 if tech_support == 'Yes' else 0
            
            # Streaming TV: No=0, Yes=1, No internet service=2
            streaming_tv = data.get('streaming_tv', 'No')
            if streaming_tv == 'No internet service':
                streaming_tv_encoded = 2
            else:
                streaming_tv_encoded = 1 if streaming_tv == 'Yes' else 0
            
            # Streaming Movies: No=0, Yes=1, No internet service=2
            streaming_movies = data.get('streaming_movies', 'No')
            if streaming_movies == 'No internet service':
                streaming_movies_encoded = 2
            else:
                streaming_movies_encoded = 1 if streaming_movies == 'Yes' else 0
            
            # Contract: Month-to-month=0, One year=1, Two year=2
            contract = data.get('contract', 'Month-to-month')
            contract_encoded = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}.get(contract, 0)
            
            # Paperless Billing: No=0, Yes=1
            paperless_billing_encoded = 1 if data.get('paperless_billing') == 'Yes' else 0
            
            # Payment Method: Electronic check=0, Mailed check=1, Bank transfer=2, Credit card=3
            payment_method = data.get('payment_method', 'Electronic check')
            payment_method_encoded = {
                'Electronic check': 0,
                'Mailed check': 1,
                'Bank transfer (automatic)': 2,
                'Credit card (automatic)': 3
            }.get(payment_method, 0)
            
            # Create feature array
            features = np.array([[
                gender_encoded,
                senior_encoded,
                partner_encoded,
                dependents_encoded,
                float(data.get('tenure', 0)),
                phone_service_encoded,
                multiple_lines_encoded,
                internet_service_encoded,
                online_security_encoded,
                online_backup_encoded,
                device_protection_encoded,
                tech_support_encoded,
                streaming_tv_encoded,
                streaming_movies_encoded,
                contract_encoded,
                paperless_billing_encoded,
                payment_method_encoded,
                float(data.get('monthly_charges', 0)),
                float(data.get('total_charges', 0))
            ]])
            
            return features
            
        except Exception as e:
            print(f"❌ Error encoding features: {e}")
            raise
    
    def assess(self, data):
        """
        Make an assessment for a single customer
        
        Args:
            data: Dictionary containing customer features
        
        Returns:
            Dictionary with assessment results
        """
        try:
            # Encode features
            features = self.encode_features(data)
            
            # Scale features if scaler is available
            if self.scaler is not None:
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = features
            
# Make assessment
            if self.model is not None:
                prediction = self.model.predict(features_scaled)[0]
                probability = self.model.predict_proba(features_scaled)[0]
                
                # Prepare result
                churn_status = 'churn' if prediction == 1 else 'stay'
                churn_probability = probability[1] if prediction == 1 else probability[0]
                confidence = max(probability) * 100
                
                # Generate recommendation
                if prediction == 1:
                    recommendation = "⚠️ This customer is at risk of churning. Consider implementing retention strategies."
                else:
                    recommendation = "✅ This customer appears satisfied with the service."
                
                result = {
                    'status': churn_status,
                    'probability': round(churn_probability * 100, 2),
                    'confidence': round(confidence, 2),
                    'assessment': int(prediction),
                    'assessment_label': 'Churn' if prediction == 1 else 'Stay',
                    'recommendation': recommendation,
                    'engine_used': self.model.__class__.__name__ if self.model else None
                }
                
                return result
            else:
                return {
                    'status': 'error',
                    'message': 'Assessment engine not loaded'
                }
                
        except Exception as e:
            print(f"❌ Assessment error: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def predict_batch(self, data_list):
        """
        Make assessments for multiple customers
        
        Args:
            data_list: List of dictionaries containing customer features
        
        Returns:
            List of assessment results
        """
        results = []
        for data in data_list:
            result = self.assess(data)
            results.append(result)
        return results


# For testing purposes
if __name__ == "__main__":
    print("="*60)
    print("🧪 Testing ChurnPredictor")
    print("="*60)
    
    # Initialize predictor
    predictor = ChurnPredictor()
    
    if predictor.model is not None:
        print("\n✅ Predictor initialized successfully")
        
        # Test with sample data
        test_data = {
            'gender': 'Male',
            'senior_citizen': 0,
            'partner': 'No',
            'dependents': 'No',
            'tenure': 12,
            'phone_service': 'Yes',
            'multiple_lines': 'No',
            'internet_service': 'DSL',
            'online_security': 'No',
            'online_backup': 'Yes',
            'device_protection': 'No',
            'tech_support': 'No',
            'streaming_tv': 'No',
            'streaming_movies': 'No',
            'contract': 'Month-to-month',
            'paperless_billing': 'Yes',
            'payment_method': 'Electronic check',
            'monthly_charges': 75.50,
            'total_charges': 906.00
        }
        
        print("\n📊 Test Data:")
        for key, value in test_data.items():
            print(f"  {key}: {value}")
        
        print("\n🔮 Making assessment...")
        result = predictor.assess(test_data)

        print("\n📊 Assessment Result:")
        for key, value in result.items():
            print(f"  {key}: {value}")
        
    else:
        print("\n❌ Failed to load predictor")
        print("Make sure model.pkl and features.pkl exist in the correct directories")
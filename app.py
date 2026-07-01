"""
RetainIQ - Customer Churn Prediction - Flask Application
End-to-End MLOps Web Application
Complete Professional Implementation
"""

import os
import sys
import json
import logging
import traceback
from datetime import datetime
from dotenv import load_dotenv

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_cors import CORS
import joblib

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-super-secret-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SESSION_TYPE'] = 'filesystem'
CORS(app)

# Global variables for models
model = None
scaler = None
feature_names = None
model_loaded = False
scaler_loaded = False

# Configuration paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.pkl')
ARTIFACTS_PATH = os.path.join(BASE_DIR, 'artifacts', 'features.pkl')

def load_models():
    """
    Load the trained model, scaler, and feature names
    """
    global model, scaler, feature_names, model_loaded, scaler_loaded
    
    logger.info("="*60)
    logger.info("Loading Models and Artifacts - RetainIQ")
    logger.info("="*60)
    
    try:
        # Load the model
        if os.path.exists(MODEL_PATH):
            logger.info(f"Loading model from: {MODEL_PATH}")
            model_data = joblib.load(MODEL_PATH)
            
            if isinstance(model_data, dict):
                model = model_data.get('model')
                scaler = model_data.get('scaler')
                feature_names = model_data.get('feature_names')
                logger.info("Model loaded successfully")
                logger.info(f"   Model Type: {model.__class__.__name__ if model else 'None'}")
                logger.info(f"   Features: {len(feature_names) if feature_names else 0}")
                model_loaded = True
            else:
                model = model_data
                logger.info("Model loaded successfully")
                logger.info(f"   Model Type: {model.__class__.__name__}")
                model_loaded = True
        else:
            logger.warning(f"Model not found at: {MODEL_PATH}")
            model_loaded = False
        
        # Load artifacts if available
        if os.path.exists(ARTIFACTS_PATH):
            logger.info(f"Loading artifacts from: {ARTIFACTS_PATH}")
            artifacts = joblib.load(ARTIFACTS_PATH)
            
            if scaler is None and 'scaler' in artifacts:
                scaler = artifacts.get('scaler')
                logger.info("Scaler loaded from artifacts")
            
            if feature_names is None and 'feature_names' in artifacts:
                feature_names = artifacts.get('feature_names')
                logger.info(f"Feature names loaded from artifacts: {len(feature_names)} features")
            
            scaler_loaded = scaler is not None
        else:
            logger.warning(f"Artifacts not found at: {ARTIFACTS_PATH}")
            scaler_loaded = False
        
        # Set default feature names if still None
        if feature_names is None:
            feature_names = [
                'gender_encoded', 'SeniorCitizen', 'Partner_encoded', 
                'Dependents_encoded', 'tenure', 'PhoneService_encoded',
                'MultipleLines_encoded', 'InternetService_encoded',
                'OnlineSecurity_encoded', 'OnlineBackup_encoded',
                'DeviceProtection_encoded', 'TechSupport_encoded',
                'StreamingTV_encoded', 'StreamingMovies_encoded',
                'Contract_encoded', 'PaperlessBilling_encoded',
                'PaymentMethod_encoded', 'MonthlyCharges', 'TotalCharges'
            ]
            logger.info("Using default feature names")
        
        logger.info("="*60)
        logger.info(f"Models Loaded: Model={model_loaded}, Scaler={scaler_loaded}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        logger.error(traceback.format_exc())
        model_loaded = False
        scaler_loaded = False
        model = None
        scaler = None

def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def encode_categorical_features(data):
    """
    Encode categorical features for prediction
    Matches the encoding used in training
    
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
        logger.error(f"Error encoding features: {e}")
        logger.error(traceback.format_exc())
        raise

def make_prediction(features):
    """
    Make prediction using the loaded model
    
    Args:
        features: numpy array of encoded features
    
    Returns:
        tuple: (prediction, probability, confidence)
    """
    try:
        # Scale features if scaler is available
        if scaler is not None:
            features_scaled = scaler.transform(features)
        else:
            features_scaled = features
        
        # Make prediction
        if model is not None:
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0]
            
            # Calculate confidence
            confidence = max(probability) * 100
            
            return prediction, probability, confidence
        else:
            return None, None, None
            
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        logger.error(traceback.format_exc())
        return None, None, None

# Load models when application starts
load_models()

# ==================== ROUTES ====================

@app.route('/')
def home():
    """
    Home page route
    """
    logger.info("Home page accessed - RetainIQ")
    return render_template('index.html', active_page='home')

@app.route('/about')
def about():
    """
    About page route
    """
    logger.info("About page accessed - RetainIQ")
    return render_template('about.html', active_page='about')

@app.route('/contact')
def contact():
    """
    Contact page route
    """
    logger.info("Contact page accessed - RetainIQ")
    return render_template('contact.html', active_page='contact')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Prediction page route
    Handles both GET and POST requests
    """
    if request.method == 'GET':
        logger.info("Prediction form page accessed - RetainIQ")
        return render_template('predict.html', active_page='predict')
    
    elif request.method == 'POST':
        logger.info("Prediction request received - RetainIQ")
        try:
            # Get form data
            form_data = {
                'gender': request.form.get('gender', 'Male'),
                'senior_citizen': safe_int(request.form.get('senior_citizen', 0)),
                'partner': request.form.get('partner', 'No'),
                'dependents': request.form.get('dependents', 'No'),
                'tenure': safe_float(request.form.get('tenure', 0)),
                'phone_service': request.form.get('phone_service', 'No'),
                'multiple_lines': request.form.get('multiple_lines', 'No'),
                'internet_service': request.form.get('internet_service', 'DSL'),
                'online_security': request.form.get('online_security', 'No'),
                'online_backup': request.form.get('online_backup', 'No'),
                'device_protection': request.form.get('device_protection', 'No'),
                'tech_support': request.form.get('tech_support', 'No'),
                'streaming_tv': request.form.get('streaming_tv', 'No'),
                'streaming_movies': request.form.get('streaming_movies', 'No'),
                'contract': request.form.get('contract', 'Month-to-month'),
                'paperless_billing': request.form.get('paperless_billing', 'No'),
                'payment_method': request.form.get('payment_method', 'Electronic check'),
                'monthly_charges': safe_float(request.form.get('monthly_charges', 0)),
                'total_charges': safe_float(request.form.get('total_charges', 0))
            }
            
            # Log form data
            logger.info(f"Form data received: {form_data}")
            
            # Encode features
            features = encode_categorical_features(form_data)
            logger.info(f"Features encoded: {features.shape}")
            
            # Make prediction
            if model is not None:
                prediction, probability, confidence = make_prediction(features)
                
                if prediction is not None:
                    # Determine churn status
                    churn_status = 'churn' if prediction == 1 else 'stay'
                    churn_probability = probability[1] if prediction == 1 else probability[0]
                    
                    # Generate recommendation
                    if prediction == 1:
                        recommendation = (
                            "This customer is at risk of churning. "
                            "Consider implementing retention strategies such as:\n"
                            "Offering discounts or promotional deals\n"
                            "Improving customer service experience\n"
                            "Providing loyalty rewards or incentives\n"
                            "Personalizing communication and offers"
                        )
                        alert_type = 'danger'
                    else:
                        recommendation = (
                            "This customer appears satisfied with the service. "
                            "Recommendations:\n"
                            "Continue providing excellent service\n"
                            "Consider offering premium features\n"
                            "Implement referral bonus programs\n"
                            "Maintain regular engagement"
                        )
                        alert_type = 'success'
                    
                    # Create result data
                    result_data = {
                        'status': churn_status,
                        'probability': round(churn_probability * 100, 2),
                        'confidence': round(confidence, 2),
                        'recommendation': recommendation,
                        'alert_type': alert_type,
                        'prediction': int(prediction),
                        'prediction_label': 'Churn' if prediction == 1 else 'Stay',
                        'model_used': model.__class__.__name__
                    }
                    
                    logger.info(f"Prediction made: {result_data['prediction_label']} (Confidence: {result_data['confidence']}%)")
                    
                    # Store in session
                    session['last_prediction'] = result_data
                    
                    # Render result page
                    return render_template('result.html', result=result_data, active_page='result')
                else:
                    logger.error("Prediction failed - model returned None")
                    return render_template('predict.html', 
                                         error='Prediction failed. Please try again.',
                                         active_page='predict')
            else:
                logger.warning("Model not loaded. Using mock prediction for demonstration.")
                
                # Mock prediction for demonstration
                mock_prediction = np.random.choice([0, 1], p=[0.7, 0.3])
                
                if mock_prediction == 1:
                    churn_status = 'churn'
                    churn_probability = np.random.uniform(0.6, 0.9)
                    recommendation = (
                        "This customer is at risk of churning. "
                        "Consider implementing retention strategies to prevent churn."
                    )
                    alert_type = 'danger'
                else:
                    churn_status = 'stay'
                    churn_probability = np.random.uniform(0.7, 0.95)
                    recommendation = (
                        "This customer appears satisfied with the service. "
                        "Continue providing excellent service."
                    )
                    alert_type = 'success'
                
                result_data = {
                    'status': churn_status,
                    'probability': round(churn_probability * 100, 2),
                    'confidence': round(churn_probability * 100, 2),
                    'recommendation': recommendation,
                    'alert_type': alert_type,
                    'prediction': int(mock_prediction),
                    'prediction_label': 'Churn' if mock_prediction == 1 else 'Stay',
                    'model_used': 'Mock (Model not loaded)'
                }
                
                session['last_prediction'] = result_data
                
                return render_template('result.html', result=result_data, active_page='result')
                
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            logger.error(traceback.format_exc())
            return render_template('predict.html', 
                                 error='An error occurred during prediction. Please try again.',
                                 active_page='predict')

@app.route('/result')
def result():
    """
    Result page route
    """
    logger.info("Result page accessed - RetainIQ")
    result_data = session.get('last_prediction', None)
    
    if result_data is None:
        logger.warning("No prediction data in session")
        return redirect(url_for('predict'))
    
    return render_template('result.html', result=result_data, active_page='result')

@app.errorhandler(404)
def not_found(error):
    """
    404 Error handler
    """
    logger.warning(f"404 Error: {request.url} - RetainIQ")
    return render_template('404.html', active_page='404'), 404

@app.errorhandler(500)
def internal_error(error):
    """
    500 Error handler
    """
    logger.error(f"500 Error: {error} - RetainIQ")
    return render_template('404.html', error='Internal Server Error', active_page='404'), 500

@app.errorhandler(405)
def method_not_allowed(error):
    """
    405 Error handler
    """
    logger.warning(f"405 Error: {request.method} {request.url} - RetainIQ")
    return render_template('404.html', error='Method Not Allowed', active_page='404'), 405

# ==================== API ENDPOINTS ====================

@app.route('/api/health')
def health_check():
    """
    Health check endpoint for deployment
    """
    return jsonify({
        'status': 'healthy',
        'application': 'RetainIQ',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'feature_count': len(feature_names) if feature_names else 0,
        'model_type': model.__class__.__name__ if model else None,
        'version': '2.0.0',
        'timestamp': datetime.now().isoformat(),
        'uptime': 'running'
    })

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint for predictions
    Accepts JSON data and returns prediction results
    """
    try:
        data = request.json
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data provided'
            }), 400
        
        logger.info(f"API Prediction request received: {data}")
        
        # Extract features
        features = encode_categorical_features(data)
        
        if model is not None:
            prediction, probability, confidence = make_prediction(features)
            
            if prediction is not None:
                return jsonify({
                    'status': 'success',
                    'application': 'RetainIQ',
                    'prediction': int(prediction),
                    'prediction_label': 'Churn' if prediction == 1 else 'Stay',
                    'probability': {
                        'churn': round(float(probability[1]), 4),
                        'stay': round(float(probability[0]), 4)
                    },
                    'confidence': round(float(confidence), 2),
                    'model_used': model.__class__.__name__
                })
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'Prediction failed'
                }), 500
        else:
            return jsonify({
                'status': 'error',
                'message': 'Model not loaded'
            }), 503
            
    except Exception as e:
        logger.error(f"API Prediction Error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/api/model/info')
def model_info():
    """
    Get model information
    """
    return jsonify({
        'application': 'RetainIQ',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'feature_names': feature_names if feature_names else [],
        'feature_count': len(feature_names) if feature_names else 0,
        'model_type': model.__class__.__name__ if model else None,
        'scaler_type': scaler.__class__.__name__ if scaler else None
    })

@app.route('/api/predict/form')
def prediction_form_schema():
    """
    Get the prediction form schema
    This helps frontend understand what fields are needed
    """
    return jsonify({
        'application': 'RetainIQ',
        'fields': [
            {'name': 'gender', 'type': 'select', 'options': ['Male', 'Female'], 'required': True},
            {'name': 'senior_citizen', 'type': 'select', 'options': [0, 1], 'required': True},
            {'name': 'partner', 'type': 'select', 'options': ['No', 'Yes'], 'required': True},
            {'name': 'dependents', 'type': 'select', 'options': ['No', 'Yes'], 'required': True},
            {'name': 'tenure', 'type': 'number', 'min': 0, 'max': 100, 'required': True},
            {'name': 'phone_service', 'type': 'select', 'options': ['No', 'Yes'], 'required': True},
            {'name': 'multiple_lines', 'type': 'select', 'options': ['No', 'Yes', 'No phone service'], 'required': True},
            {'name': 'internet_service', 'type': 'select', 'options': ['DSL', 'Fiber optic', 'No'], 'required': True},
            {'name': 'online_security', 'type': 'select', 'options': ['No', 'Yes', 'No internet service'], 'required': True},
            {'name': 'online_backup', 'type': 'select', 'options': ['No', 'Yes', 'No internet service'], 'required': True},
            {'name': 'device_protection', 'type': 'select', 'options': ['No', 'Yes', 'No internet service'], 'required': True},
            {'name': 'tech_support', 'type': 'select', 'options': ['No', 'Yes', 'No internet service'], 'required': True},
            {'name': 'streaming_tv', 'type': 'select', 'options': ['No', 'Yes', 'No internet service'], 'required': True},
            {'name': 'streaming_movies', 'type': 'select', 'options': ['No', 'Yes', 'No internet service'], 'required': True},
            {'name': 'contract', 'type': 'select', 'options': ['Month-to-month', 'One year', 'Two year'], 'required': True},
            {'name': 'paperless_billing', 'type': 'select', 'options': ['No', 'Yes'], 'required': True},
            {'name': 'payment_method', 'type': 'select', 'options': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], 'required': True},
            {'name': 'monthly_charges', 'type': 'number', 'min': 0, 'step': 0.01, 'required': True},
            {'name': 'total_charges', 'type': 'number', 'min': 0, 'step': 0.01, 'required': True}
        ]
    })

# ==================== RUN APPLICATION ====================

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV', 'development') == 'development'
    
    logger.info("="*60)
    logger.info("Starting RetainIQ - Customer Churn Prediction Application")
    logger.info(f"Server will run on: http://localhost:{port}")
    logger.info(f"Debug mode: {debug}")
    logger.info("="*60)
    
    app.run(debug=debug, host='0.0.0.0', port=port)
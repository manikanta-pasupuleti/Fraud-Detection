"""
Real-time Credit Card Fraud Prediction Application
Uses a pre-trained machine learning model to predict fraudulent transactions.
"""

import streamlit as st
import joblib
import pandas as pd
from functools import lru_cache


# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

TRANSACTION_TYPES = ['purchase', 'refund']
LOCATIONS = [
    'Dallas', 'Houston', 'Los Angeles', 'New York',
    'Philadelphia', 'Phoenix', 'San Antonio',
    'San Diego', 'San Jose'
]

NUMERICAL_FEATURES = ['Amount', 'MerchantID', 'TransactionHour']

FEATURE_COLUMNS = [
    'Amount',
    'MerchantID',
    'TransactionHour',
    'TransactionType_refund',
    'Location_Dallas',
    'Location_Houston',
    'Location_Los Angeles',
    'Location_New York',
    'Location_Philadelphia',
    'Location_Phoenix',
    'Location_San Antonio',
    'Location_San Diego',
    'Location_San Jose'
]

LOCATION_FEATURES = [f'Location_{city}' for city in LOCATIONS]


# ============================================================================
# MODEL LOADING (CACHED)
# ============================================================================

@lru_cache(maxsize=1)
def load_model():
    """Load pre-trained fraud detection model."""
    try:
        return joblib.load('best_fraud_detection_model.joblib')
    except FileNotFoundError:
        st.error("❌ Error: Model file 'best_fraud_detection_model.joblib' not found.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()


@lru_cache(maxsize=1)
def load_scaler():
    """Load StandardScaler for feature normalization."""
    try:
        return joblib.load('scaler.joblib')
    except FileNotFoundError:
        st.error("❌ Error: Scaler file 'scaler.joblib' not found.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading scaler: {str(e)}")
        st.stop()


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def preprocess_transaction(amount, merchant_id, transaction_hour, transaction_type, location):
    """
    Process raw transaction inputs into model-ready features.
    
    Args:
        amount: Transaction amount
        merchant_id: Merchant identifier
        transaction_hour: Hour of transaction (0-23)
        transaction_type: 'purchase' or 'refund'
        location: City name
    
    Returns:
        DataFrame with processed features ready for model prediction
    """
    # Create initial DataFrame
    input_df = pd.DataFrame({
        'Amount': [amount],
        'MerchantID': [merchant_id],
        'TransactionHour': [transaction_hour],
        'TransactionType': [transaction_type],
        'Location': [location]
    })
    
    # One-hot encode TransactionType (refund indicator)
    input_df['TransactionType_refund'] = (input_df['TransactionType'] == 'refund').astype(int)
    
    # One-hot encode Location
    for location_feature in LOCATION_FEATURES:
        city_name = location_feature.replace('Location_', '')
        input_df[location_feature] = (input_df['Location'] == city_name).astype(int)
    
    # Drop original categorical columns (no longer needed)
    input_df = input_df.drop(columns=['TransactionType', 'Location'])
    
    # Scale numerical features
    scaler = load_scaler()
    input_df[NUMERICAL_FEATURES] = scaler.transform(input_df[NUMERICAL_FEATURES])
    
    # Ensure all expected columns are present in correct order
    for col in FEATURE_COLUMNS:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder columns to match training data
    return input_df[FEATURE_COLUMNS]


# ============================================================================
# UI AND PAGE SETUP
# ============================================================================

st.set_page_config(page_title="Fraud Detection", page_icon="🚀", layout="wide")

st.write("🚀 App started successfully")
st.title('Real-time Credit Card Fraud Prediction')

st.write("""
This application uses a pre-trained machine learning model to predict fraudulent credit card transactions in real-time. 
Input the transaction details below to get a fraud prediction.
""")


# ============================================================================
# USER INPUT SECTION
# ============================================================================

st.sidebar.header('Transaction Details')

amount = st.sidebar.number_input(
    'Amount ($)',
    min_value=0.01,
    max_value=1000.00,
    value=500.00,
    step=0.01
)

merchant_id = st.sidebar.number_input(
    'Merchant ID',
    min_value=1,
    max_value=10000,
    value=5000,
    step=1
)

transaction_hour = st.sidebar.number_input(
    'Transaction Hour (0-23)',
    min_value=0,
    max_value=23,
    value=12,
    step=1
)

transaction_type = st.sidebar.selectbox(
    'Transaction Type',
    options=TRANSACTION_TYPES,
    index=0
)

location = st.sidebar.selectbox(
    'Location',
    options=LOCATIONS,
    index=0
)


# ============================================================================
# PREDICTION SECTION
# ============================================================================

if st.sidebar.button('Predict Fraud', key='predict_fraud_button'):
    try:
        # Preprocess transaction
        processed_input = preprocess_transaction(
            amount, merchant_id, transaction_hour, transaction_type, location
        )
        
        # Load model and make prediction
        model = load_model()
        prediction = model.predict(processed_input)[0]
        prediction_proba = model.predict_proba(processed_input)[0]
        
        # Display results
        st.write("### Prediction Results:")
        
        if prediction == 1:
            st.error("#### ⚠️ Fraudulent Transaction Detected!")
            fraud_probability = prediction_proba[1] * 100
            st.write(f"##### Probability of Fraud: {fraud_probability:.2f}%")
        else:
            st.success("#### ✅ Non-Fraudulent Transaction")
            legitimate_probability = prediction_proba[0] * 100
            st.write(f"##### Probability of Legitimate Transaction: {legitimate_probability:.2f}%")
        
        # Disclaimer
        st.info(
            "⚠️ **Disclaimer**: This prediction is based on a machine learning model and should be used "
            "as a decision-support tool, not as a final authority."
        )
        
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")



import streamlit as st
st.write("🚀 App started successfully")


st.title('Real-time Credit Card Fraud Prediction')
st.write("""
This application uses a pre-trained machine learning model to predict fraudulent credit card transactions in real-time. 
Input the transaction details below to get a fraud prediction.
""")
import joblib

# Load the trained model
model = joblib.load('best_fraud_detection_model.joblib')

# Define options for categorical features based on original dataset
transaction_type_options = ['purchase', 'refund'] # Inferred from one-hot encoding
location_options = [
    'Dallas',
    'Houston',
    'Los Angeles',
    'New York',
    'Philadelphia',
    'Phoenix',
    'San Antonio',
    'San Diego',
    'San Jose'
]
# Inferred from one-hot encoding

# --- User Input Fields ---
st.sidebar.header('Transaction Details')

amount = st.sidebar.number_input('Amount', min_value=0.01, max_value=1000.00, value=500.00, step=0.01)
merchant_id = st.sidebar.number_input('MerchantID', min_value=1, max_value=10000, value=5000, step=1)
transaction_hour = st.sidebar.number_input('TransactionHour (0-23)', min_value=0, max_value=23, value=12, step=1)
transaction_type = st.sidebar.selectbox(
    'TransactionType',
    options=transaction_type_options,
    index=0,
    key='transaction_type_select'
)

location = st.sidebar.selectbox(
    'Location',
    options=location_options,
    index=0,
    key='location_select'
)


# Load the StandardScaler
scaler = joblib.load('scaler.joblib')

import pandas as pd

# Create a DataFrame from user inputs
input_data = {
    'Amount': [amount],
    'MerchantID': [merchant_id],
    'TransactionHour': [transaction_hour],
    'TransactionType': [transaction_type],
    'Location': [location]
}
input_df = pd.DataFrame(input_data)

#st.write("### Raw Input Data:")
#st.write(input_df)

# Manually perform one-hot encoding for categorical features
# TransactionType columns based on X_train.columns after drop_first=True for TransactionType
input_df['TransactionType_refund'] = (input_df['TransactionType'] == 'refund').astype(int)

# One-hot encode Location (EXACT training features)
location_cols = [
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

for col in location_cols:
    input_df[col] = (input_df['Location'] == col.replace('Location_', '')).astype(int)

#st.write("### After One-Hot Encoding:")
#st.write(input_df)

# List of numerical features to scale
numerical_features = ['Amount', 'MerchantID', 'TransactionHour']

# Apply StandardScaler to the numerical features
input_df[numerical_features] = scaler.transform(input_df[numerical_features])

#st.write("### After Scaling Numerical Features:")
#st.write(input_df)

# Define the expected column order based on X_train.columns from training
# This needs to match the exact columns and order after one-hot encoding and dropping 'TransactionID', 'TransactionDate', 'IsFraud'
feature_columns = [
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


# Ensure all expected columns are present, fill with False if not (for location columns not selected)
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = False # Default to False if a specific OHE column is not present

# Reorder columns to match the training data
processed_input = input_df[feature_columns]

#st.write("### Final Processed Input for Model:")
#st.write(processed_input)

# Make prediction when 'Predict' button is clicked
if st.sidebar.button('Predict Fraud',key='predict_fraud_button'):
    # Make prediction
    prediction = model.predict(processed_input)
    prediction_proba = model.predict_proba(processed_input)

    st.write("### Prediction Results:")
    if prediction[0] == 1:
        st.error("#### Fraudulent Transaction Detected!")
        st.write(f"##### Probability of Fraud: {prediction_proba[0][1]*100:.2f}%")
    else:
        st.success("#### Non-Fraudulent Transaction")
        st.write(f"##### Probability of Non-Fraud: {prediction_proba[0][0]*100:.2f}%")
    st.info(
        "⚠️ This prediction is based on a machine learning model and should be used "
        "as a decision-support tool, not as a final authority."
    )


# Define options for categorical features based on original dataset
transaction_type_options = ['purchase', 'refund'] # Updated based on user instruction
location_options = [
    'Dallas', 'Houston', 'Los Angeles', 'New York',
    'Philadelphia', 'Phoenix', 'San Antonio',
    'San Diego', 'San Jose'
] # Updated based on user instruction and X.columns



# Manually perform one-hot encoding for categorical features
# TransactionType columns based on X_train.columns after drop_first=True for TransactionType
input_df['TransactionType_refund'] = (input_df['TransactionType'] == 'refund').astype(bool)
# Note: 'TransactionType_withdrawal' is not present in X.columns, so it's not included

# Location columns - all are present in X_train.columns, implying no drop_first for Location
location_cols = [
    'Location_Dallas', 'Location_Houston', 'Location_Los Angeles', 'Location_New York',
    'Location_Philadelphia', 'Location_Phoenix', 'Location_San Antonio',
    'Location_San Diego', 'Location_San Jose'
] # Updated to match X.columns

for col in location_cols:
    input_df[col] = (input_df['Location'] == col.replace('Location_', '')).astype(bool)

#st.write("### After One-Hot Encoding:")
#st.write(input_df)

# List of numerical features to scale
numerical_features = ['Amount', 'MerchantID', 'TransactionHour']

# Apply StandardScaler to the numerical features
input_df[numerical_features] = scaler.transform(input_df[numerical_features])

#st.write("### After Scaling Numerical Features:")
#st.write(input_df)

# Define the expected column order based on X_train.columns from training
# This needs to match the exact columns and order after one-hot encoding and dropping 'TransactionID', 'TransactionDate', 'IsFraud'
feature_columns = [
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
] # Updated to exactly match X.columns

# Ensure all expected columns are present, fill with False if not (for location columns not selected)
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = False # Default to False if a specific OHE column is not present

# Reorder columns to match the training data
processed_input = input_df[feature_columns]

#st.write("### Final Processed Input for Model:")
#st.write(processed_input)

# Define options for categorical features based on original dataset
transaction_type_options = ['purchase', 'refund'] # Updated based on user instruction
location_options = [
    'Dallas', 'Houston', 'Los Angeles', 'New York',
    'Philadelphia', 'Phoenix', 'San Antonio',
    'San Diego', 'San Jose'
]

# Manually perform one-hot encoding for categorical features
# TransactionType columns based on X_train.columns after drop_first=True for TransactionType
input_df['TransactionType_refund'] = (input_df['TransactionType'] == 'refund').astype(int)
# Note: 'TransactionType_withdrawal' is not present in X.columns, so it's not included

# Location columns - all are present in X_train.columns, implying no drop_first for Location
location_cols = [
    'Location_Dallas', 'Location_Houston', 'Location_Los Angeles', 'Location_New York',
    'Location_Philadelphia', 'Location_Phoenix', 'Location_San Antonio',
    'Location_San Diego', 'Location_San Jose'
] # Updated to match X.columns

for col in location_cols:
    input_df[col] = (input_df['Location'] == col.replace('Location_', '')).astype(int)

# Drop original categorical columns
input_df = input_df.drop(columns=['TransactionType', 'Location'])

#st.write("### After One-Hot Encoding:")
#st.write(input_df)

# List of numerical features to scale
numerical_features = ['Amount', 'MerchantID', 'TransactionHour']

# Apply StandardScaler to the numerical features
input_df[numerical_features] = scaler.transform(input_df[numerical_features])

#st.write("### After Scaling Numerical Features:")
#st.write(input_df)

# Define the expected column order based on X_train.columns from training
# This needs to match the exact columns and order after one-hot encoding and dropping 'TransactionID', 'TransactionDate', 'IsFraud'
feature_columns = [
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
] # Updated to exactly match X.columns

# Ensure all expected columns are present, fill with 0 if not (for location columns not selected, to match astype(int))
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0 # Default to 0 if a specific OHE column is not present

# Reorder columns to match the training data
processed_input = input_df[feature_columns]

#st.write("### Final Processed Input for Model:")
#st.write(processed_input)

#st.write("Model expects:", len(feature_columns), "features")
#st.write("App provides:", processed_input.shape[1], "features")



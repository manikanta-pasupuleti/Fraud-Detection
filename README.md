# Real-time Credit Card Fraud Detection

A machine learning-powered Streamlit application that predicts fraudulent credit card transactions in real-time.

## Features

- **Real-time Fraud Detection**: Input transaction details and get instant fraud predictions
- **Pre-trained ML Model**: Uses a logistic regression classifier trained on historical fraud data
- **User-friendly Interface**: Simple sidebar controls for easy transaction input
- **Probability Scores**: Shows confidence level for fraud predictions
- **Scalable Input Processing**: Handles feature scaling and encoding automatically

## Project Structure

```
fraud_detection_app/
├── app.py                              # Main Streamlit application
├── best_fraud_detection_model.joblib   # Pre-trained logistic regression model
├── scaler.joblib                       # StandardScaler for feature normalization
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/manikanta-pasupuleti/Fraud-Detection.git
   cd Fraud-Detection
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **Access the application**
   - Open your browser and navigate to `http://localhost:8501`

3. **Make a prediction**
   - Fill in the transaction details in the left sidebar:
     - **Amount**: Transaction amount (0.01 - 1000.00)
     - **MerchantID**: Merchant identifier (1 - 10000)
     - **TransactionHour**: Hour of transaction (0 - 23)
     - **TransactionType**: Choose between 'purchase' or 'refund'
     - **Location**: Select from 9 US cities (Dallas, Houston, Los Angeles, New York, Philadelphia, Phoenix, San Antonio, San Diego, San Jose)
   - Click the **Predict Fraud** button to get results

## Model Details

- **Algorithm**: Logistic Regression
- **Features**: 13 features
  - 3 numerical: Amount, MerchantID, TransactionHour
  - 1 categorical: TransactionType (refund indicator)
  - 9 categorical: Location (city one-hot encoded)
- **Preprocessing**:
  - StandardScaler normalization for numerical features
  - One-hot encoding for categorical features
  - Features are scaled and encoded automatically by the app

## Requirements

- streamlit
- pandas
- numpy
- scikit-learn
- joblib
- xgboost
- imbalanced-learn
- matplotlib
- seaborn

See `requirements.txt` for specific versions.

## Important Notes

⚠️ **Disclaimer**: This prediction is based on a machine learning model and should be used as a decision-support tool, not as a final authority on fraud detection.

## How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is open source and available under the MIT License.

## Contact

For questions or issues, please visit the [GitHub repository](https://github.com/manikanta-pasupuleti/Fraud-Detection) or open an issue.

---

**Happy Fraud Detection! 🚀**

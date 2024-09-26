from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import psycopg2
import os
import uvicorn
import logging
from datetime import datetime

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the saved model and scaler
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, 'best_model.pkl')
    scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    logger.info("Model and scaler loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

# Database connection parameters (Use environment variables)
db_params = {
    'host': 'branchhomeworkdb.cv8nj4hg6yra.ap-south-1.rds.amazonaws.com',
    'port': '5432',
    'database': 'branchdsprojectgps',
    'user': 'datascientist',
    'password': '47eyYBLT0laW5j9U24Uuy8gLcrN'
}

def fetch_user_data(user_id):
    """
    Fetches user attributes, loan application data, and GPS fixes from the database.
    Computes necessary features for prediction.
    """
    conn = None
    try:
        # Establish database connection
        conn = psycopg2.connect(**db_params)

        # Fetch user_attributes
        user_attributes_query = """
        SELECT age, cash_incoming_30days
        FROM user_attributes WHERE user_id = %s;
        """
        user_attributes = pd.read_sql_query(user_attributes_query, conn, params=(user_id,))

        if user_attributes.empty:
            logger.error(f"No user attributes found for user_id: {user_id}")
            return None

        # Fetch loan application data to extract application_dayofweek and application_hour
        loan_outcomes_query = """
        SELECT application_at
        FROM loan_outcomes WHERE user_id = %s;
        """
        loan_data = pd.read_sql_query(loan_outcomes_query, conn, params=(user_id,))

        if loan_data.empty:
            logger.error(f"No loan data found for user_id: {user_id}")
            return None

        # Convert application_at to datetime and extract features
        loan_data['application_at'] = pd.to_datetime(loan_data['application_at'])
        user_attributes['application_dayofweek'] = loan_data['application_at'].dt.dayofweek  # Monday=0, Sunday=6
        user_attributes['application_hour'] = loan_data['application_at'].dt.hour

        # Fetch gps_fixes data to compute gps_fix_count and avg_accuracy
        gps_fixes_query = """
        SELECT accuracy
        FROM gps_fixes WHERE user_id = %s;
        """
        gps_data = pd.read_sql_query(gps_fixes_query, conn, params=(user_id,))

        if gps_data.empty:
            logger.error(f"No GPS data found for user_id: {user_id}")
            # Set default values or handle as needed
            user_attributes['gps_fix_count'] = 0
            user_attributes['avg_accuracy'] = None  # Or a default value
        else:
            user_attributes['gps_fix_count'] = len(gps_data)
            user_attributes['avg_accuracy'] = gps_data['accuracy'].mean()

        return user_attributes

    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return None

    finally:
        # Close the connection
        if conn:
            conn.close()

class UserID(BaseModel):
    user_id: str

@app.post('/predict_loan_outcome')
def predict_loan_outcome(user: UserID):
    """
    Predicts the loan outcome based on the user_id.
    """
    if model is None or scaler is None:
        logger.error("Model or scaler not loaded.")
        raise HTTPException(status_code=500, detail="Model or scaler not loaded.")

    # Fetch data from the database
    input_data = fetch_user_data(user.user_id)
    
    if input_data is None:
        logger.error(f"User data not found for user_id: {user.user_id}")
        raise HTTPException(status_code=404, detail="User data not found.")

    # Feature selection
    features = ['age', 'cash_incoming_30days', 'gps_fix_count', 'avg_accuracy', 'application_dayofweek', 'application_hour']

    # Check if all required features are present
    missing_features = [feature for feature in features if feature not in input_data.columns]
    if missing_features:
        logger.error(f"Incomplete data for prediction. Missing features: {missing_features}")
        raise HTTPException(status_code=500, detail=f"Incomplete data for prediction. Missing features: {missing_features}")

    # Handle missing values (if any)
    input_data.fillna(0, inplace=True)  # Or use a more appropriate method

    # Scale the input data
    scaled_input = scaler.transform(input_data[features])

    # Make the prediction
    try:
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1]
    except Exception as e:
        logger.error(f"Error during model prediction: {e}")
        raise HTTPException(status_code=500, detail="Error during model prediction.")

    outcome = 'repaid' if prediction == 1 else 'defaulted'

    logger.info(f"Prediction made for user_id {user.user_id}: {outcome}, Probability of repayment: {probability:.2f}")

    return {
        'user_id': user.user_id,
        'prediction': outcome,
        'probability_of_repayment': probability
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

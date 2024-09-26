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

db_params = {
    'host': '',
    'port': '',
    'database': '',
    'user': '',
    'password': ''
}

def fetch_user_data(user_id):
    """
    Fetches user attributes, loan application data, and GPS fixes from the database.
    Computes necessary features for prediction.
    """
    conn = None
    try:
        conn = psycopg2.connect(**db_params)

        user_attributes_query = """
        SELECT age, cash_incoming_30days
        FROM user_attributes WHERE user_id = %s;
        """
        user_attributes = pd.read_sql_query(user_attributes_query, conn, params=(user_id,))

        if user_attributes.empty:
            logger.error(f"No user attributes found for user_id: {user_id}")
            return None

        loan_outcomes_query = """
        SELECT application_at
        FROM loan_outcomes WHERE user_id = %s;
        """
        loan_data = pd.read_sql_query(loan_outcomes_query, conn, params=(user_id,))

        if loan_data.empty:
            logger.error(f"No loan data found for user_id: {user_id}")
            return None

        loan_data['application_at'] = pd.to_datetime(loan_data['application_at'])
        user_attributes['application_dayofweek'] = loan_data['application_at'].dt.dayofweek  
        user_attributes['application_hour'] = loan_data['application_at'].dt.hour

        gps_fixes_query = """
        SELECT accuracy
        FROM gps_fixes WHERE user_id = %s;
        """
        gps_data = pd.read_sql_query(gps_fixes_query, conn, params=(user_id,))

        if gps_data.empty:
            logger.error(f"No GPS data found for user_id: {user_id}")
            user_attributes['gps_fix_count'] = 0
            user_attributes['avg_accuracy'] = None  
        else:
            user_attributes['gps_fix_count'] = len(gps_data)
            user_attributes['avg_accuracy'] = gps_data['accuracy'].mean()

        return user_attributes

    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return None

    finally:
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

    input_data = fetch_user_data(user.user_id)
    
    if input_data is None:
        logger.error(f"User data not found for user_id: {user.user_id}")
        raise HTTPException(status_code=404, detail="User data not found.")

    features = ['age', 'cash_incoming_30days', 'gps_fix_count', 'avg_accuracy', 'application_dayofweek', 'application_hour']

    missing_features = [feature for feature in features if feature not in input_data.columns]
    if missing_features:
        logger.error(f"Incomplete data for prediction. Missing features: {missing_features}")
        raise HTTPException(status_code=500, detail=f"Incomplete data for prediction. Missing features: {missing_features}")

    input_data.fillna(0, inplace=True)  

    scaled_input = scaler.transform(input_data[features])

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

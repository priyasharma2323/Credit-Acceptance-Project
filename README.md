# Loan Outcome Prediction API

This API is built using FastAPI and is designed to predict the outcome of a loan application based on a user's attributes, loan application data, and GPS data. It fetches the necessary features from a PostgreSQL database and uses a pre-trained model to make predictions.

## Features:
- **Predict Loan Outcome**: Based on user data such as age, cash inflow, GPS fix count, and time of application.
- **Database Integration**: Fetches user data from a PostgreSQL database.
- **Model Integration**: Uses a pre-trained machine learning model (`best_model.pkl`) and scaler (`scaler.pkl`) to make predictions.

## Requirements:
- Python 3.8+
- FastAPI
- Uvicorn
- Joblib
- Pandas
- Psycopg2

## Installation:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/loan-outcome-api.git
   cd loan-outcome-api
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the environment variables:
   ```bash
   export DB_HOST
   export DB_PORT
   export DB_NAME
   export DB_USER
   export DB_PASSWORD
   ```

4. Ensure that the `best_model.pkl` and `scaler.pkl` files are in the project directory.

## Running the API:

To start the FastAPI server, run:
```bash
uvicorn main:app --reload
```

The API will be available at `http://0.0.0.0:8000`.

## API Endpoints:

### 1. Predict Loan Outcome

- **Endpoint**: `/predict_loan_outcome`
- **Method**: POST
- **Request Body**: 
  - `user_id` (string): The unique identifier of the user.
  
- **Response**: 
  - `user_id` (string): The unique identifier of the user.
  - `prediction` (string): The predicted loan outcome (`repaid` or `defaulted`).
  - `probability_of_repayment` (float): The probability that the user will repay the loan.
  
- **Example Request**:
  ```bash
  curl -X 'POST' \
    'http://0.0.0.0:8000/predict_loan_outcome' \
    -H 'Content-Type: application/json' \
    -d '{"user_id": "12345"}'
  ```

- **Example Response**:
  ```json
  {
    "user_id": "12345",
    "prediction": "repaid",
    "probability_of_repayment": 0.85
  }
  ```

## Database Configuration:
The API connects to a PostgreSQL database. It fetches user attributes, loan application data, and GPS data from the following tables:
- `user_attributes`: Contains user details like age and cash inflow.
- `loan_outcomes`: Contains loan application timestamp.
- `gps_fixes`: Contains GPS accuracy data.

Modify the `db_params` in the script or use environment variables to set the database connection parameters.

## Logging:
The API uses Python's built-in logging module to log key events, such as:
- Successful loading of the model and scaler.
- Errors in database queries or model predictions.
- Predictions made for users.

## Error Handling:
- **500 Internal Server Error**: If the model or scaler fails to load, or if there is an issue during prediction.
- **404 Not Found**: If no user data is found for the given user ID.
- **500 Internal Server Error**: If any of the required features are missing or if the prediction fails due to incomplete data.

## License:
This project is licensed under the MIT License.

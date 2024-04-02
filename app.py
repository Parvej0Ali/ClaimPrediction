from flask import Flask, request, jsonify
import joblib

# Load the model
loaded_model = joblib.load('D:\\Parvej Ali\\Lumiq\\Bootcamp\\EPIC 7\\models\\rf_prediction_model.joblib')

# Create a Flask app
app = Flask(__name__)

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    input_data = request.get_json()

    # Ensure all required input fields are present
    required_fields = ['DeductibleAmtPaid', 'ClmProcedureCode_1', 'ClmProcedureCode_2',
                       'ClmProcedureCode_3', 'ClmProcedureCode_4', 'ClmProcedureCode_5',
                       'Gender', 'Race', 'State', 'County', 'NoOfMonths_PartACov',
                       'NoOfMonths_PartBCov', 'ChronicCond_Alzheimer',
                       'ChronicCond_Heartfailure', 'ChronicCond_KidneyDisease',
                       'ChronicCond_Cancer', 'ChronicCond_ObstrPulmonary',
                       'ChronicCond_Depression', 'ChronicCond_Diabetes',
                       'ChronicCond_IschemicHeart', 'ChronicCond_Osteoporasis',
                       'ChronicCond_rheumatoidarthritis', 'ChronicCond_stroke',
                       'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt',
                       'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt']
    for field in required_fields:
        if field not in input_data:
            return jsonify({'error': f'Missing required field: {field}'}), 400

    # Extract input features from the data
    features = [input_data[field] for field in required_fields]

    # Make a prediction using the loaded model
    prediction = loaded_model.predict([features])[0]

    # Return the prediction as JSON
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)

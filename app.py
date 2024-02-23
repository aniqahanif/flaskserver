from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model and scaler
with open('svm_model.pkl', 'rb') as model_file:
    svm_model = pickle.load(model_file)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/predict', methods=['POST'])
def predict():
    # Extract information from request
    data = request.json
    audio_file_name = data['audio_file']
    education = data['education']  # Assuming you need this for something else
    age = data['age']  # Assuming you need this for something else
    gender=data['gender']
    result=0

    # Load test.csv and find the row for the audio file
    df_test = pd.read_csv('test_data.csv')
    row = df_test[df_test['ID'] == audio_file_name]

    if row.empty:
        return jsonify({'hello here, prediction': str(result)})

    # Preprocess the row (remove unnecessary columns and scale the features)
    features = row.drop(['ID', 'Audio_File', 'Audio File','language_English','language_German','language_Greek','language_Spanish','diagnosis_HC','diagnosis_AD'], axis=1)

    # Make a prediction
    prediction = svm_model.predict(features)
    result = prediction  # Assuming binary classification
    print("called")
    # Return the prediction result
    return jsonify({'hello here, prediction': str(result)})

if __name__ == '__main__':
    app.run(debug=True,port=5000)

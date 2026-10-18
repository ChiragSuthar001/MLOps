from flask import Flask, request, jsonify, render_template
import joblib
import tensorflow as tf
import keras
import numpy as np
import os

app = Flask(__name__, static_folder='statics')

# Load the TensorFlow model
model  = None
encoder = None
scaler = None

model = keras.models.load_model('my_nn_iris_model.keras')
encoder = joblib.load('iris_encoder.pkl')
scaler = joblib.load('iris_scaler.pkl')

_inputs = ('sepal_length', 'sepal_width', 'petal_length', 'petal_width')

def _getPrediction(data):
    sepal_length = float(data['sepal_length'])
    sepal_width = float(data['sepal_width'])
    petal_length = float(data['petal_length'])
    petal_width = float(data['petal_width'])

    # Perform the prediction
    input_data = np.array([sepal_length, sepal_width, petal_length, petal_width])[np.newaxis, ]

    print("====== INPUT DATA ======")
    print(input_data)
    print("========================")

    input_data_scaled = scaler.transform(input_data)

    prediction = model.predict(input_data_scaled)

    predicted_classes_onehot = np.zeros_like(prediction)
    predicted_classes_onehot[np.arange(len(prediction)), np.argmax(prediction, axis=1)] = 1

    predicted_class = encoder.inverse_transform(predicted_classes_onehot)

    return jsonify({"predicted_class": predicted_class[0][0]})

@app.route('/')
def home():
    return "Welcome to the Iris Classifier API!"

@app.route("/health")
def health_check():
    return jsonify({"status": "healthy"})

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.get_json(silent=True) if request.is_json else request.form
            if not data:
                return jsonify({"error": "Request body must contain prediction inputs."}), 400
            return _getPrediction(data)
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    elif request.method == 'GET':
        if (not all(i in request.args for i in _inputs)):
            return render_template('predict.html')
        data = request.args
        return _getPrediction(data)
    else:
        return "Unsupported HTTP method"

if __name__ == "__main__":
    model_file = "my_nn_iris_model.keras"
    encoder_file = "iris_encoder.pkl"
    scaler_file = "iris_scaler.pkl"

    required_files = [model_file, encoder_file, scaler_file]

    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(
                f"Required asset missing: '{file}'. Please ensure the file is in the correct directory."
            )

    print("Loading models and preprocessors...")
    model = keras.models.load_model(model_file)
    encoder = joblib.load(encoder_file)
    scaler = joblib.load(scaler_file)
    print("Models and preprocessors loaded successfully.")

    app.run(debug=True, host='0.0.0.0', port=3000)

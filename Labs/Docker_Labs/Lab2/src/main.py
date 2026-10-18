from flask import Flask, request, jsonify, render_template
import joblib
import tensorflow as tf
import keras
import numpy as np

app = Flask(__name__, static_folder='statics')

# URL of your Flask API for making predictions
api_url = 'http://0.0.0.0:3000/predict'

# Load the TensorFlow model
model = keras.models.load_model('my_nn_iris_model.keras')
encoder = joblib.load('iris_encoder.pkl')
scaler = joblib.load('iris_scaler.pkl')

class_labels = ['Setosa', 'Versicolor', 'Virginica']


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
            data = request.form
            sepal_length = float(data['sepal_length'])
            sepal_width = float(data['sepal_width'])
            petal_length = float(data['petal_length'])
            petal_width = float(data['petal_width'])

            # Perform the prediction
            input_data = np.array([sepal_length, sepal_width, petal_length, petal_width])[np.newaxis, ]

            input_data_scaled = scaler.transform(input_data)

            prediction = model.predict(input_data_scaled)

            predicted_classes_onehot = np.zeros_like(prediction)
            predicted_classes_onehot[np.arange(len(prediction)), np.argmax(prediction, axis=1)] = 1

            predicted_class = encoder.inverse_transform(predicted_classes_onehot)

            return jsonify({"predicted_class": predicted_class[0][0]})
        except Exception as e:
            return jsonify({"error": str(e)})
    elif request.method == 'GET':
        return render_template('predict.html')
    else:
        return "Unsupported HTTP method"

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=4000)

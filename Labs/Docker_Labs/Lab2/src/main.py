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

_inputs = (
    'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
    'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
    'proanthocyanins', 'color_intensity', 'hue',
    'od280_od315_of_diluted_wines', 'proline'
)

_outputs = {
    'class_0': 'Barolo',
    'class_1': 'Grignolino',
    'class_2': 'Barbera'
}

def _getPrediction(data):
    feature_values = [float(data[name]) for name in _inputs]

    # Perform the prediction
    input_data = np.array(feature_values)[np.newaxis, ]

    debug = os.getenv('DEBUG', 'False').lower() in ('true', 't')
    if debug:
        print("====== INPUT DATA ======")
        print(input_data)
        print("========================")

    input_data_scaled = scaler.transform(input_data)

    prediction = model.predict(input_data_scaled)

    predicted_classes_onehot = np.zeros_like(prediction)
    predicted_classes_onehot[np.arange(len(prediction)), np.argmax(prediction, axis=1)] = 1

    predicted_class = encoder.inverse_transform(predicted_classes_onehot)

    return jsonify({"predicted_class": _outputs[predicted_class[0][0]]})

@app.route('/')
def home():
    return "Welcome to the Wine Classifier API!"

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
    model_file = "my_nn_wine_model.keras"
    encoder_file = "wine_encoder.pkl"
    scaler_file = "wine_scaler.pkl"

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

    app.run(host='0.0.0.0', port=os.getenv('PORT', 3000))

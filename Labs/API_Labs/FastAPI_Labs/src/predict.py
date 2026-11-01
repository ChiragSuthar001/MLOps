import joblib
import keras
import os
import numpy as np

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

def predict_data(X):
    """
    Predict the class labels for the input data.
    Args:
        X (numpy.ndarray): Input data for which predictions are to be made.
    Returns:
        y_pred (numpy.ndarray): Predicted class labels.
    """
    model_file = "../model/my_nn_wine_model.keras"
    encoder_file = "../model/wine_encoder.pkl"
    scaler_file = "../model/wine_scaler.pkl"

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

    data = X.model_dump()
    feature_values = [float(data[name]) for name in _inputs]

    # Perform the prediction
    input_data = np.array(feature_values)[np.newaxis, ]

    input_data_scaled = scaler.transform(input_data)

    pred = model.predict(input_data_scaled)

    predicted_classes_onehot = np.zeros_like(pred)
    predicted_classes_onehot[np.arange(len(pred)), np.argmax(pred, axis=1)] = 1

    predicted_class = encoder.inverse_transform(predicted_classes_onehot)

    return _outputs[predicted_class[0][0]]
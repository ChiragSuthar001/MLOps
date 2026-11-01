from keras.models import Sequential
from keras.layers import Dense, Input
import joblib
from data import load_data, split_data
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def fit_model(X_train, y_train):
    """
    Train a neural network and save the model to a file.
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values.
    """
    model = Sequential([
        # Using Relu as the activation function, and input layer with 4 features
        Input(shape=(13,)),
        Dense(32, activation='relu'), 
        Dense(16, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(X_train, y_train, epochs=50, batch_size=8, validation_split=0.1, verbose=1)
    model.save('../model/my_nn_wine_model.keras')

if __name__ == "__main__":
    X, y = load_data()
    encoder = OneHotEncoder(sparse_output=False)
    scaler = StandardScaler()

    X_train, X_test, y_train, y_test = split_data(X, y)

    y_train_encoded = encoder.fit_transform(y_train)
    X_train_scaled = scaler.fit_transform(X_train)

    joblib.dump(scaler, '../model/wine_scaler.pkl')
    joblib.dump(encoder, '../model/wine_encoder.pkl')
    
    fit_model(X_train_scaled, y_train_encoded)

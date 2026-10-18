import joblib
from keras.models import Sequential
from keras.layers import Dense, Input
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if __name__ == '__main__': 
    # Working with the Wine Dataset.
    wine = load_wine()
    X = wine.data
    y = wine.target_names[wine.target].reshape(-1, 1)

    #Preprocess data: One-Hot Encode the labels (0, 1, 2 -> arrays of length 3)
    encoder = OneHotEncoder(sparse_output=False)

    scaler = StandardScaler()

    # Splitting the data into 80-20 train-test data sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)

    print(f"Features shape: {X.shape}, Labels shape: {y_train_encoded.shape}")
    print(f"First 5 samples of features:\n{X[:5]} with corresponding labels:\n{y_train_encoded[:5]}")

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # The Neural Network model
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

    print("Starting training of the neural network model...")
    model.fit(X_train_scaled, y_train_encoded, epochs=100, batch_size=8, validation_split=0.1, verbose=1)

    # 6. Evaluate accuracy on unseen test data
    loss, accuracy = model.evaluate(X_test_scaled, y_test_encoded, verbose=0)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

    joblib.dump(scaler, 'wine_scaler.pkl')
    print("Scaler was saved as 'wine_scaler.pkl'")

    joblib.dump(encoder, 'wine_encoder.pkl')
    print("Encoder was saved as 'wine_encoder.pkl'")

    model.save('my_nn_wine_model.keras')
    print("Model was trained and saved")

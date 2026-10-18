import joblib
from keras.models import Sequential
from keras.layers import Dense, Input
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if __name__ == '__main__': 
    # Working with the IRIS Dataset.
    iris = load_iris()
    X = iris.data
    y = iris.target_names[iris.target].reshape(-1, 1)

    #Preprocess data: One-Hot Encode the labels (0, 1, 2 -> arrays of length 3)
    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y)

    scaler = StandardScaler()

    print(f"Features shape: {X.shape}, Labels shape: {y_encoded.shape}")
    print(f"First 5 samples of features:\n{X[:5]} with corresponding labels:\n{y_encoded[:5]}")

    # Splitting the data into 80-20 train-test data sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # The Neural Network model
    model = Sequential([
        # Using Relu as the activation function, and input layer with 4 features
        Input(shape=(4,)),
        Dense(10, activation='relu'), 
        Dense(8, activation='relu'),
        Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Starting training of the neural network model...")
    model.fit(X_train_scaled, y_train, epochs=100, batch_size=5, verbose=1)

    # 6. Evaluate accuracy on unseen test data
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

    joblib.dump(scaler, 'iris_scaler.pkl')
    print("Scaler was saved as 'iris_scaler.pkl'")

    joblib.dump(encoder, 'iris_encoder.pkl')
    print("Encoder was saved as 'iris_encoder.pkl'")

    model.save('my_nn_iris_model.keras')
    print("Model was trained and saved")

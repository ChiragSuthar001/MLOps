# Docker Lab 2 — Iris Classifier using Tensorflow (Docker)

This document walks you thought the changes and how to start up the project

## Training ([src/model_training.py](https://github.com/ChiragSuthar001/MLOps/blob/main/Labs/Docker_Labs/Lab2/src/model_training.py))

The training script uses scikit-learn's built-in Iris dataset. It one-hot encodes the
labels, splits the data into 80% training and 20% test data, standardizes the features,
and trains a three-class neural network with Keras for 100 epochs.

I am saving three things at the end of the model training

- `Model` -> The trained model with the final weights saved as `'my_nn_iris_model.keras'`
- `Scaler` -> The Standard Scaler used during training to be used during inference to ensure accurate rescaling during inference.
- `Encoder` -> The one hot encoder, so that during inference we can dynamically get the class name without relying on the same class sequence during inference. To reduce chances of error.

### Run training locally

From the project root, install the dependencies and run the training script:

```bash
python3 -m pip install -r requirements.txt
python3 src/model_training.py
```

The script evaluates the model on the test set and writes these files to the project
root:

- `my_nn_iris_model.keras` - the trained Keras model
- `iris_encoder.pkl` - the fitted label encoder
- `iris_scaler.pkl` - the fitted feature scaler

These artifacts are loaded by `src/main.py` when the prediction API starts.

### Train as part of the Docker build

The first stage of the `dockerfile` installs the dependencies and runs
`model_training.py` automatically. Build the image from the project root:

```bash
docker build -f dockerfile -t iris-classifier .
```

The generated model, encoder, and scaler are copied into the serving stage of the image, so no separate training command is required before starting the container, and the model works out of the box for the final container image.

## Flask API ([src/main.py](https://github.com/ChiragSuthar001/MLOps/blob/main/Labs/Docker_Labs/Lab2/src/main.py))

- Loads `my_nn_iris_model.keras`, `iris_scaler.pkl` and `iris_encoder.pkl` results from the model training at start up. If any of these files are missing it would throw an exception at the start time.
- `GET /health` returns the health of the server
- `GET /predict` with empty body returns the HTML page for browser viewing.
- `GET /predict` with request args returns JSON response.
- `POST /predict` with request form data or json, returns json response for requests from browser.

## Web UI (src/template/)

- Used the default `predict.html` - just changed the switch to match my input to the html file.

# Use of Docker

Used docker to multi-stage build to make a light weight final image with weights of the trained model which can be used to out of the box directly without needing another training.

## Mutli-stage build:
- **Stage: `model_training`** trains the model while only copying the needed files for training. `src/model_training.py` and `requirements.txt`.
- **Stage: `serving`** - the Final stage in the dockerfile which by the rule of docker 'the final image is the last reachable FROM stage'. It only contains the code that runs the model trained in previous stage. It copies the three files model files created in first stage and then serves them through `main.py`.
- **`Expose: 80`** exposes the port 80

Used `.dockerignore` to exclude all the files and folders not necessary for the build and remove noise from rebuilds.
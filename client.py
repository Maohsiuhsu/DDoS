"""
This program is a Flower client for decentralized training in federated learning architecture.
Main functions include:
1. Load the Edge-IIoT dataset and preprocess it (standardization and OneHot encoding)
2. Establish a multi-layer perceptron (MLP) model for classification (for industrial IoT attack types)
3. Training and evaluation on the local side
4. Supports whether to use float16 precision when selecting the training or transfer model weights
"""
import flwr as fl
import numpy as np
import tensorflow as tf
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser(description="Flower Client Options")
parser.add_argument("--train_dtype", choices=["float32", "float16"], default="float32", help="訓練使用的資料精度")
parser.add_argument("--send_dtype", choices=["float32", "float16"], default="float32", help="上傳權重使用的精度")
args = parser.parse_args()


physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU is available and memory growth is set.")
    except RuntimeError as e:
        print(f"Error setting up GPU: {e}")
else:
    print("No GPU found. Running on CPU.")

# load data 
csv_file = fr"/home/root001/Howard/flower/ddos_flower/Dataset/edgeiiot/Edge-IIoTset dataset/classifition/fortrain_data_E.csv"
df = pd.read_csv(csv_file, low_memory=False)
df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

X = df.drop(columns=["Attack_type", "Attack_label", "Attack_category"]).values
Y = df["Attack_type"].values

np.random.seed(10)
idx = np.random.permutation(len(X))
X, Y = X[idx], Y[idx]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.astype(args.train_dtype))
X_test = scaler.transform(X_test.astype(args.train_dtype))

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
Y_train = encoder.fit_transform(Y_train.reshape(-1, 1))
Y_test = encoder.transform(Y_test.reshape(-1, 1))

# define model 

def create_model():
    model = Sequential([
        Dense(128, input_shape=(X_train.shape[1],), activation='relu'),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(len(np.unique(Y)), activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), metrics=['accuracy'])
    return model

# calculate metrics

def compute_metrics(y_true, y_pred, class_labels):
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    report = classification_report(y_true_labels, y_pred_labels, output_dict=True, zero_division=0)
    metrics = {
        "accuracy": report["accuracy"],
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "f1-score": report["macro avg"]["f1-score"]
    }
    for i, label in enumerate(class_labels):
        if str(i) in report:
            metrics.update({
                f"{label}_precision": report[str(i)]["precision"],
                f"{label}_recall": report[str(i)]["recall"],
                f"{label}_f1": report[str(i)]["f1-score"]
            })
    return metrics

# Flower client

model = create_model()

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [w.astype(args.send_dtype) for w in model.get_weights()]

    def fit(self, parameters, config):
        model.set_weights(parameters)
        history = model.fit(X_train, Y_train, epochs=1, batch_size=4096, verbose=1)
        loss = float(history.history["loss"][-1])
        accuracy = float(history.history["accuracy"][-1])
        y_pred = model.predict(X_train)
        metrics = compute_metrics(Y_train, y_pred, encoder.categories_[0])
        metrics.update({"loss": loss, "accuracy": accuracy})
        return [w.astype(args.send_dtype) for w in model.get_weights()], len(X_train), metrics

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        y_pred = model.predict(X_test)
        metrics = compute_metrics(Y_test, y_pred, encoder.categories_[0])
        loss = model.evaluate(X_test, Y_test, verbose=0)[0]
        print(f"Client Evaluation - Loss: {loss:.4f}, Accuracy: {metrics['accuracy']:.4f}, "
              f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, "
              f"F1-Score: {metrics['f1-score']:.4f}")
        return float(loss), len(X_test), metrics

fl.client.start_numpy_client(server_address="localhost:8080", client=FlowerClient())


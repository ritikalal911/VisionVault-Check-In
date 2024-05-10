import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from src.traning.architecture import Architecture
from keras.callbacks import Callback
from insightface.app import FaceAnalysis
from keras.callbacks import EarlyStopping
import cv2

class ProgressCallback(Callback):
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs

    def on_epoch_end(self, epoch, logs=None):
        epochs = (epoch + 1)/ self.epochs
        print(f"Epoch {epoch + 1}/{self.epochs}")
        print(f"Loss: {logs['loss']}, Acc: {logs['accuracy']}")
        return logs, epochs

def load_embeddings(embeddings_path):
    with open(embeddings_path, 'rb') as f:
        embeddings_data = pickle.load(f)
    return embeddings_data

def generate_face_embedding(embeddings_data):
    print("[INFO] Extracting embeddings...")
    known_embeddings = []
    known_names = []

    for name, emb_list in embeddings_data.items():
        known_embeddings.extend(emb_list)
        known_names.extend([name] * len(emb_list))

    print(len(known_embeddings), "embeddings extracted")

    data = {"embeddings": np.array(known_embeddings), "names": known_names}
    return data

def train_model(embeddings, arguments):
    data = embeddings
    labels = data["names"]
    embeddings = data["embeddings"]

    # Use label encoder to convert string labels to numeric values
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    num_classes = len(np.unique(labels))

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

    # Define input shape
    input_shape = X_train.shape[1]

    # Build the softmax model
    arc = Architecture(input_shape=(input_shape,), num_classes=num_classes)
    model = arc.build_model()

    # Train the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    progress_callback = ProgressCallback(epochs=50)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=8, callbacks=[progress_callback,early_stopping])

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    train_accuracy = history.history['accuracy'][-1]
    val_accuracy = history.history['val_accuracy'][-1]
    print("Test Accuracy:", test_accuracy)
    print("Train Accuracy:", train_accuracy)
    print("Validation Accuracy:", val_accuracy)

    # Save the trained face recognition model
    model.save(arguments['model'])
    label_encoder_file = open(arguments["le"], "wb")
    label_encoder_file.write(pickle.dumps(label_encoder))
    label_encoder_file.close()
    print("Training Complete. Model trained seamlessly on fresh data.")

    return train_accuracy, val_accuracy, test_accuracy

if __name__ == "__main__":
    embeddings_path = "src/models/embeddings.pickle"
    arguments = {
        "model": "src/models/face_recognition_model.h5",
        "le": "src/models/label_encoder.pickle"
    }

    # Load embeddings
    embeddings_data = load_embeddings(embeddings_path)

    # Generate face embeddings
    embeddings = generate_face_embedding(embeddings_data)

    # Train model
    train_model(embeddings, arguments)

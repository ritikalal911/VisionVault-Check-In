import os
import cv2
import numpy as np
import pickle
import dlib
from insightface.app import FaceAnalysis
from keras.models import load_model

def load_label_encoder(le_path):
    with open(le_path, 'rb') as f:
        label_encoder = pickle.load(f)
    return label_encoder

def load_embeddings(embeddings_path):
    with open(embeddings_path, 'rb') as f:
        embeddings_data = pickle.load(f)
    
    embeddings = []
    labels = []

    for label, emb_list in embeddings_data.items():
        embeddings.extend(emb_list)
        labels.extend([label] * len(emb_list))

    return np.array(embeddings), np.array(labels)


def predict_faces(face_analysis, label_encoder, model, embeddings, threshold=0.70):
    cap = cv2.VideoCapture(0)  # Open webcam
    detector = dlib.get_frontal_face_detector()  # Initialize dlib face detector
    predictions = []

    while True:
        ret, frame = cap.read()  # Read frame from webcam
        if not ret:
            break

        # Convert frame to grayscale for dlib face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform face detection using dlib
        faces = detector(gray)
        if faces:
            for face in faces:
                # Get face bounding box coordinates
                x, y, w, h = face.left(), face.top(), face.width(), face.height()

                # Extract face region from the frame
                face_img = frame[y:y+h, x:x+w]

                # Perform face embedding using insightface
                embedding = face_analysis.get(face_img)
                if embedding:
                    embedding = embedding[0]['embedding']
                    embedding = np.expand_dims(embedding, axis=0)
                    prediction = model.predict(embedding)
                    max_prob = np.max(prediction)
                    if max_prob > threshold:  # Check if prediction accuracy is greater than threshold
                        predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
                        confidence_level = f"{max_prob * 100:.2f}%"
                    else:
                        predicted_label = 'Unknown'
                        confidence_level = 'N/A'

                    predictions.append(predicted_label)

                    # Draw rectangle around the face and display predicted label and confidence level
                    text = f"{predicted_label} ({confidence_level})"
                    cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                    cv2.putText(frame, text, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Display the frame
        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    # Paths to model, label encoder, and embeddings
    model_path = "src/models/face_recognition_model.h5"
    le_path = "src/models/label_encoder.pickle"
    embeddings_path = "src/models/embeddings.pickle"

    # Load label encoder
    label_encoder = load_label_encoder(le_path)

    # Load face recognition model
    model = load_model(model_path)

    # Load embeddings
    embeddings, _ = load_embeddings(embeddings_path)

    # Initialize FaceAnalysis for face detection and recognition
    face_analysis = FaceAnalysis(name='buffalo_s')
    face_analysis.prepare(ctx_id=1, det_size=(256, 256))

    # Perform real-time face recognition
    predict_faces(face_analysis, label_encoder, model, embeddings)

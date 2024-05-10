import tkinter
from tkinter import messagebox, Text
import customtkinter
from PIL import Image, ImageTk, ImageSequence
import os
import numpy as np
from datetime import datetime
import time
from keras.models import load_model
from insightface.app import FaceAnalysis

from src.predictor.prediction import load_label_encoder, load_embeddings, predict_faces

customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("blue")

class DetectionFrame(customtkinter.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)

        # Project description or under development message
        dev_message = "This feature is under development."
        dev_label = customtkinter.CTkLabel(self, text=dev_message, font=("Arial", 14))
        dev_label.grid(row=0, column=0, columnspan=3, padx=(20, 20), pady=(20, 20), sticky="nsew")

        # Add GIF image
        # gif_path = "src\images\working-on.gif"  # Replace with the path to your GIF file
        # gif_image = Image.open(gif_path)
        # gif_image = gif_image.resize((200, 200), Image.ANTIALIAS)
        # gif_photo = ImageTk.PhotoImage(gif_image)

        # gif_label = customtkinter.CTkLabel(self, image=gif_photo)
        # gif_label.image = gif_photo
        # gif_label.grid(row=0, column=0, columnspan=3, padx=(20, 20), pady=(20, 20), sticky="nsew")

        # Add detect button and output text area
        self.detect_button = customtkinter.CTkButton(self, text="Detect", command=self.detect_face)
        self.detect_button.grid(row=1, column=0, columnspan=3, padx=(20, 20), pady=(20, 20), sticky="nsew")

        self.print_output_text = Text(self, wrap="word", height=22.4, state="disabled", bg='#343638', fg='white')
        self.print_output_text.grid(row=2, column=0, columnspan=3, padx=(20, 20), pady=(20, 20), sticky="nsew")

    def detect_face(self):
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

if __name__ == "__main__":
    root = customtkinter.CTk()
    root.title("VisionVault")
    root.geometry("1100x580")
    root.resizable(False, False)
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(0, weight=1)

    DetectionFrame(root).grid(row=0, column=0, sticky="nsew")

    root.mainloop()



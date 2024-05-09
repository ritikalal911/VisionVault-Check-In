import tkinter as tk
from tkinter import Text
import customtkinter
from customtkinter import CTkProgressBar
import os
import random
import threading
from PIL import Image, ImageTk
from src.face_embedding.face_embedding import generate_embeddings
from src.traning.train import load_embeddings, generate_face_embedding, train_model, ProgressCallback

customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("blue")

class ModelTrainFrame(customtkinter.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)
        self.grid_columnconfigure(3, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.grid_rowconfigure(3, weight=1)
        self.grid_rowconfigure(4, weight=1)
        self.grid_rowconfigure(5, weight=1)

        project_text = "You can train the model by clicking the 'Train Model' button. The model will be trained using the embeddings extracted from the images. The training process may take some time. Once the model is trained successfully, you will see a message indicating the same." 
        
        project_label = customtkinter.CTkLabel(self, text=project_text, wraplength=1000, justify='center', font=("Arial", 15))
        project_label.grid(row=0, column=0, columnspan=4, padx=(20, 20), pady=(20, 0), sticky="nsew")

        side_image_bar = customtkinter.CTkFrame(self)
        side_image_bar.grid(row=1, column=0, rowspan=5, padx=(20, 20), pady=(20, 20), sticky="nsew")
        
        
        self.train_button = customtkinter.CTkButton(self, text="Train Model", command=self.train_model)
        self.train_button.grid(row=1, column=1, columnspan=3, padx=(20, 20), pady=(20, 20), sticky="nsew")

        self.progress_bar = CTkProgressBar(self, width=100, height=20)
        self.progress_bar.set(0)
        self.progress_bar.grid(row=2, column=1, columnspan=3, padx=(20, 20), pady=(20, 20), sticky="nsew")

        self.print_output_text = Text(self, wrap="word", height=22.4, state="disabled",bg='#343638', fg='white')
        self.print_output_text.grid(row=3,rowspan=2 ,column=1, columnspan=3, padx=(20, 20), pady=(20, 20), sticky="nsew")

    def update_progress(self, value):
        self.progress_bar.set(value)

    def train_model(self):
        # self.add_random_images_to_table()
        def start_training():
            data_folder = "data"
            output_pickle_file = os.path.join("src", "Models", "embeddings.pickle")

            def success_callback(subfolder):
                status = f"Embedding of {subfolder} is successful."
                self.print_output_text.config(state="normal")
                self.print_output_text.insert("end", status + "\n")
                self.print_output_text.config(state="disabled")
                self.update()
        
            generate_embeddings(data_folder, output_pickle_file, success_callback)
            self.print_output_text.config(state="normal")
            self.print_output_text.insert("end", "Embeddings generated successfully!\n")
            self.print_output_text.config(state="disabled")
            self.update()
            
            embeddings_data = load_embeddings(output_pickle_file)
            data = generate_face_embedding(embeddings_data)
            self.print_output_text.config(state="normal")
            self.print_output_text.insert("end", "Embeddings extracted successfully!\n")
            self.print_output_text.config(state="disabled")
            self.update()

            arguments = {
                "model": "src/models/face_recognition_model.h5",
                "le": "src/models/label_encoder.pickle"
            }
            self.print_output_text.config(state="normal")
            self.print_output_text.insert("end", "Training model...\n")
            self.print_output_text.config(state="disabled")
            self.update()
            train_model(data,arguments)
            self.progress_bar.set(1)  # Set progress bar to 100 after training completes
            self.print_output_text.config(state="normal")
            self.print_output_text.insert("end", "Model trained successfully!\n")
            self.print_output_text.config(state="disabled",bg='#009966', fg='white')
            self.update()

        # Start training in a separate thread
        train_thread = threading.Thread(target=start_training)
        train_thread.start()

        # Update progress bar value gradually after every 2 seconds
        def update_progress_bar(value=0):
            if value <= 0.8:
                self.update_progress(value)
                value += 0.01
                self.after(1000, update_progress_bar, value)
            elif value > 0.8 and value < 0.90:
                self.update_progress(value)
                value += 0.01
                self.after(300, update_progress_bar, value)
        update_progress_bar()


        

if __name__ == "__main__":
    root = customtkinter.CTk()
    root.title("VisionVault")
    root.geometry("1100x580")
    root.resizable(False, False)
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(0, weight=1)
    stop_event = threading.Event()  # Create a threading Event object
    model_train_frame = ModelTrainFrame(root)
    model_train_frame.grid(row=0, column=0, sticky="nsew")
    root.mainloop()

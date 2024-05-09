import tkinter
from tkinter import messagebox, Text
import customtkinter
from PIL import Image, ImageTk
import os
import numpy as np
from datetime import datetime
import time

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

        self.train_button = customtkinter.CTkButton(self, text="Train Model", command=self.train_model)
        self.train_button.grid(row=0, column=0, columnspan=3, padx=(20, 20), pady=(20, 20), sticky="nsew")

        self.print_output_text = Text(self, wrap="word", height=22.4, state="disabled", bg='#343638', fg='white')
        self.print_output_text.grid(row=1, column=0, columnspan=3, padx=(20, 20), pady=(20, 20), sticky="nsew")

    def train_model(self):
        for i in range(0, 100, 10):
            self.print_output_text.config(state="normal")
            self.print_output_text.delete("1.0", "end")  # Clear previous content
            self.print_output_text.insert("end", f"Training model...{i}%\n")
            self.print_output_text.config(state="disabled")
            self.update()  # Update the display
            time.sleep(1)
        self.print_output_text.config(state="normal")
        self.print_output_text.delete("1.0", "end")  # Clear previous content
        self.print_output_text.insert("end", "Model trained successfully!\n")
        self.print_output_text.config(state="disabled")

if __name__ == "__main__":
    root = customtkinter.CTk()
    root.title("VisionVault")
    root.geometry("1100x580")
    root.resizable(False, False)
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(0, weight=1)

    model_train_frame = ModelTrainFrame(root)
    model_train_frame.grid(row=0, column=0, sticky="nsew")

    root.mainloop()

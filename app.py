import tkinter as tk
from tkinter import messagebox, Text
import customtkinter
from PIL import Image, ImageTk
import os
import numpy as np
from datetime import datetime
import time
import dlib
import threading

from src.uiFiles.getFaceUI import GetFaceFrame
from src.uiFiles.modelTrainUi import ModelTrainFrame
from src.uiFiles.prdictionUi import DetectionFrame

customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("blue")

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("VisionVault-Check-In")
        self.geometry("1100x580")

        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure((0, 1), weight=1)

        self.tabview = customtkinter.CTkTabview(self, width=400)
        self.tabview.grid(row=0, column=0, columnspan=2, rowspan=3, padx=(20, 20), pady=(20, 20), sticky="nsew")

        Collect_data_tab = self.tabview.add("Collect Data")
        self.setup_collect_data_tab(Collect_data_tab)
        
        Train_tab = self.tabview.add("Train")
        self.setup_train_tab(Train_tab)

        attendance_details_tab = self.tabview.add("Attendance Details")
        self.setup_attendance_details_tab(attendance_details_tab)

    def setup_collect_data_tab(self, tab):
        # Instantiate GetFaceFrame and pass the tab
        get_face_frame = GetFaceFrame(tab)
        get_face_frame.pack(expand=True, fill="both")
        
    def setup_train_tab(self, tab):
        model_train_frame = ModelTrainFrame(tab)
        model_train_frame.pack(expand=True, fill="both")

    def setup_attendance_details_tab(self, tab):
        DetectionFrame(tab).pack(expand=True, fill="both")
        # pass

def start_app():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    thread = threading.Thread(target=start_app)
    thread.start()

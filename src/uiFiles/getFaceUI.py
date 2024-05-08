import cv2
import tkinter as tk
from tkinter import messagebox,Text
import customtkinter
from PIL import Image, ImageTk
import os
import numpy as np
from datetime import datetime
import time
import dlib

customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("blue")

class GetFaceFrame(customtkinter.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        # self.args = args
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("src\Models\shape_predictor_68_face_landmarks.dat")
        # super().__init__(self)

        # self.title("VisionVault-Check-In")
        # self.geometry("1100x580")

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)

        self.camera_label = tk.Label(self, bg="black")
        self.camera_label.grid(row=0, column=0,columnspan=2,padx=(20, 20), pady=(20, 20), sticky="nsew")

        Frame = customtkinter.CTkFrame(self)
        Frame.grid(row=0, column=2, padx=(20, 0), pady=(20, 20), sticky="nsew")

        Frame.grid_columnconfigure(0, weight=1)
        Frame.grid_columnconfigure(1, weight=1)

        name_label = customtkinter.CTkLabel(Frame, text="Name:")
        name_label.grid(row=0, column=0, padx=10, pady=10, sticky="e")

        self.name_entry = customtkinter.CTkEntry(Frame)
        self.name_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        enrollment_label = customtkinter.CTkLabel(Frame, text="Enrollment No:")
        enrollment_label.grid(row=1, column=0, padx=10, pady=10, sticky="e")

        self.enrollment_entry = customtkinter.CTkEntry(Frame)
        self.enrollment_entry.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

        submit_button = customtkinter.CTkButton(Frame, text="Submit", command=self.submit_attendance)
        submit_button.grid(row=2, column=0, columnspan=2, pady=20, sticky="nsew")

        # Add a Text widget to display real-time terminal output
        self.print_output_text = Text(Frame, wrap="word", height=22.4, state="disabled",bg='#343638', fg='white')
        self.print_output_text.grid(row=3, column=0,columnspan=2, pady=(20, 0), sticky="nsew")

        # Initialize the VideoCapture object

        # Set the desired frame width and height
        frame_width = 640  # Adjust the width as needed
        frame_height = 580  # Adjust the height as needed


        self.capture = cv2.VideoCapture(0)
        # Set the frame width and height
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(os.path.join("src", "Models", "shape_predictor_68_face_landmarks.dat"))
        self.update_camera()


    def submit_attendance(self):

        enrollment_no = self.enrollment_entry.get()
        name = self.name_entry.get()
        count = 0
        faces = 0
        frames = 0

        folder_path = os.path.join("data", enrollment_no)
        os.makedirs(folder_path, exist_ok=True)

        while count < 50:
            ret, frame = self.capture.read()
            frames += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_rect = self.detector(gray)

            if len(faces_rect) != 0:
                face_rect = faces_rect[0]

                landmarks = self.predictor(gray, face_rect)
                landmarks = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(68)])

                cv2.rectangle(frame, (face_rect.left()-5, face_rect.top()-25), (face_rect.right()+5, face_rect.bottom()+5), (0, 255, 0), 2)

                cropped_face = frame[face_rect.top()-20:face_rect.bottom(), face_rect.left():face_rect.right()]
                cropped_face_resized = cv2.resize(cropped_face, (112, 112))
                save_path = os.path.join(folder_path, f"{enrollment_no}_{count+1}.png")
                cv2.imwrite(save_path,cropped_face_resized)

                for point in landmarks:
                    cv2.circle(frame, tuple(point), 2, (0, 155, 255), 2)

                count += 1

                # Display real-time print output
                print_output = f"Collected Frame : {count}"
                self.print_output_text.config(state="normal")
                self.print_output_text.insert("end", print_output + "\n")
                self.print_output_text.config(state="disabled")
                # Auto-scroll to the bottom
                self.print_output_text.yview("end")

                # Calculate padding to center the image
                # label_width = self.camera_label.winfo_width()
                # label_height = self.camera_label.winfo_height()
                # image_width = frame.width()
                # image_height = frame.height()

                # padx = max((label_width - image_width) // 2, 0)
                # pady = max((label_height - image_height) // 2, 0)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = ImageTk.PhotoImage(frame)
                self.camera_label.configure(image=frame)
                self.camera_label.image = frame
                # self.camera_label.grid_configure(padx=(padx, padx), pady=(pady, pady))
                self.update_idletasks()
                self.update()
                time.sleep(0.1)

        messagebox.showinfo("Capture Complete", "50 photos captured and saved successfully!")

    def update_camera(self):
        ret, frame = self.capture.read()
        if ret:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = ImageTk.PhotoImage(frame)
            self.camera_label.configure(image=frame)
            self.camera_label.image = frame
            self.after(10, self.update_camera)

if __name__ == "__main__":
    root = tk.Tk()
    app = GetFaceFrame(root)
    app.pack(expand=True, fill="both")
    app.mainloop()

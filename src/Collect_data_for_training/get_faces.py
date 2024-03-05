import cv2
import tkinter as tk
from tkinter import messagebox
import customtkinter
from PIL import Image, ImageTk
import os
import numpy as np
from datetime import datetime
import time
import dlib

customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("blue")

class App(customtkinter.CTk):
    def __init__(self):
        # self.args = args
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("src\Models\shape_predictor_68_face_landmarks.dat")
        super().__init__()

        self.title("VisionVault-Check-In")
        self.geometry("1100x580")

        self.grid_columnconfigure((0, 1, 2), weight=1)
        self.grid_rowconfigure((0, 1, 2), weight=1)
        

        self.camera_label = tk.Label(self, bg="black")
        self.camera_label.grid(row=0, column=0, columnspan=2,rowspan=3, padx=(20, 0), pady=(20, 20), sticky="nsew")

        self.tabview = customtkinter.CTkTabview(self, width=400)
        self.tabview.grid(row=0, column=3, rowspan=3, padx=(20, 20), pady=(20, 20), sticky="nsew")

        Collect_data_tab = self.tabview.add("Collect Data")
        self.setup_collect_data_tab(Collect_data_tab)

        Train_tab = self.tabview.add("Train")
        self.setup_train_tab(Train_tab)

        attendance_details_tab = self.tabview.add("Attendance Details")
        self.setup_attendance_details_tab(attendance_details_tab)

        self.capture = cv2.VideoCapture(0)
        # desired_width = 1050  # Adjust this to your preferred width
        # self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(os.path.join("src", "Models", "shape_predictor_68_face_landmarks.dat"))
        self.update_camera()

    def setup_collect_data_tab(self, tab):

        tab.grid_columnconfigure(0, weight=1)
        tab.grid_columnconfigure(1, weight=1)

        name_label = customtkinter.CTkLabel(tab, text="Name:")
        name_label.grid(row=0, column=0, padx=10, pady=10, sticky="e")

        self.name_entry = customtkinter.CTkEntry(tab)
        self.name_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        enrollment_label = customtkinter.CTkLabel(tab, text="Enrollment No:")
        enrollment_label.grid(row=1, column=0, padx=10, pady=10, sticky="e")

        self.enrollment_entry = customtkinter.CTkEntry(tab)
        self.enrollment_entry.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

        submit_button = customtkinter.CTkButton(tab, text="Submit", command=self.submit_attendance)
        submit_button.grid(row=2, column=0, columnspan=2, pady=20, sticky="nsew")

    def setup_attendance_details_tab(self, tab):
        pass

    def setup_train_tab(self, tab):
        pass

    def submit_attendance(self):

        enrollment_no = self.enrollment_entry.get()
        name = self.name_entry.get()
        count = 0
        faces = 0
        frames = 0
        # max_faces = int(self.args['faces'])

        folder_path = os.path.join("data", enrollment_no)
        os.makedirs(folder_path, exist_ok=True)

        while count < 50:
            ret, frame = self.capture.read()
            frames += 1

            # Convert the image to grayscale for dlib face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the grayscale image
            faces_rect = self.detector(gray)

            if len(faces_rect) != 0:
                # Assuming one face per frame
                face_rect = faces_rect[0]

                # Get facial landmarks
                landmarks = self.predictor(gray, face_rect)
                landmarks = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(68)])

                # Extract aligned face
                # nimg = self.align_face(frame, landmarks)

                # cv2.imwrite(self.args['output'] + '/' + datastring + '.jpg', nimg)
                cv2.rectangle(frame, (face_rect.left()-5, face_rect.top()-25), (face_rect.right()+5, face_rect.bottom()+5), (0, 255, 0), 2)
                # Crop the region inside the rectangle
                cropped_face = frame[face_rect.top()-20:face_rect.bottom(), face_rect.left():face_rect.right()]

                # Resize the cropped face to 112x112
                cropped_face_resized = cv2.resize(cropped_face, (112, 112))
                save_path = os.path.join(folder_path, f"{enrollment_no}_{count+1}.png")
                cv2.imwrite(save_path,cropped_face_resized)
                # cv2.imwrite(folder_path + '/' + datastring + '.jpg', cropped_face_resized)
                for point in landmarks:
                    cv2.circle(frame, tuple(point), 2, (0, 155, 255), 2)

                print(f"Faces: {faces} Frames: {frames}")
                count += 1
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = ImageTk.PhotoImage(frame)
                self.camera_label.configure(image=frame)
                self.camera_label.image = frame
                self.update_idletasks()
                self.update()
                time.sleep(0.1)


        messagebox.showinfo("Capture Complete", "50 photos captured and saved successfully!")


    # def align_face(self, frame, landmarks):
    #     # You can implement your own face alignment logic based on landmarks
    #     # This example uses a simple crop around the eyes and resize
    #     left_eye = landmarks[36:42]
    #     right_eye = landmarks[42:48]

    #     # Calculate the center of both eyes
    #     left_eye_center = np.mean(left_eye, axis=0).astype(int)
    #     right_eye_center = np.mean(right_eye, axis=0).astype(int)

    #     # Calculate the angle between the eyes
    #     angle = np.degrees(np.arctan2(right_eye_center[1] - left_eye_center[1], right_eye_center[0] - left_eye_center[0]))

    #     # Rotate the image around the center of the eyes
    #     h, w = frame.shape[:2]
    #     M = cv2.getRotationMatrix2D(tuple(left_eye_center), angle, scale=1)
    #     aligned_face = cv2.warpAffine(frame, M, (w, h))

    #     return aligned_face

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
    app = App()
    app.mainloop()

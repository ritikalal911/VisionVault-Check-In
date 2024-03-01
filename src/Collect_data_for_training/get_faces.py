import cv2
import dlib
import numpy as np
from datetime import datetime
import os
import time

class DataTraining:

    def __init__(self, args):
        self.args = args
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat")

    def get_face(self):
        cap = cv2.VideoCapture(0)

        faces = 0
        frames = 0
        max_faces = int(self.args['faces'])

        if not (os.path.exists(self.args['output'])):
            os.makedirs(self.args['output'])

        while faces < max_faces:
            ret, frame = cap.read()
            frames += 1

            datastring = str(datetime.now().microsecond)

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
                cv2.imwrite(self.args['output'] + '/' + datastring + '.jpg', cropped_face_resized)
                for point in landmarks:
                    cv2.circle(frame, tuple(point), 2, (0, 155, 255), 2)

                print(f"Faces: {faces} Frames: {frames}")
                faces += 1
                time.sleep(0.05)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"Total faces: {faces} Total frames: {frames}")

    def align_face(self, frame, landmarks):
        # You can implement your own face alignment logic based on landmarks
        # This example uses a simple crop around the eyes and resize
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]

        # Calculate the center of both eyes
        left_eye_center = np.mean(left_eye, axis=0).astype(int)
        right_eye_center = np.mean(right_eye, axis=0).astype(int)

        # Calculate the angle between the eyes
        angle = np.degrees(np.arctan2(right_eye_center[1] - left_eye_center[1], right_eye_center[0] - left_eye_center[0]))

        # Rotate the image around the center of the eyes
        h, w = frame.shape[:2]
        M = cv2.getRotationMatrix2D(tuple(left_eye_center), angle, scale=1)
        aligned_face = cv2.warpAffine(frame, M, (w, h))

        return aligned_face

if __name__ == "__main__":
    args = {'output': 'data/test', 'faces': 50}
    dt = DataTraining(args)
    dt.get_face()

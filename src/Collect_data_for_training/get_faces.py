import sys
import os
import cv2
import numpy as np
from datetime import datetime
from mtcnn.mtcnn import MTCNN
from src.insightface.src.common import face_preprocess

sys.path.append('../insightface/deploy')
sys.path.append('../insightface/src/common')

class DataTraining:

    def __init__(self,args):
        self.args = args
        self.detector = MTCNN()

    def get_face(self):
        cap = cv2.VideoCapture(0)

        faces = 0
        frames = 0
        max_faces = int(self.args['faces'])
        max_bbox = np.zeros(4)

        if not (os.path.exists(self.args['output'])):
            os.makedirs(self.args['output'])

        while faces<max_faces:
            ret,frame = cap.read()
            frames += 1

            datastring = str(datetime.now().microsecond)
            # here I am using the MTCNN to detect faces or get the bounding box of the face
            bboxes = self.detector.detect_faces(frame)

            if len(bboxes) != 0:
                
                max_area = 0
                for bboxe in bboxes:
                    bbox = bboxe['box']
                    bbox = np.array([bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]])
                    keypoints = bboxe['keypoints']
                    area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])  
                    if area > max_area:
                        max_area = area
                        max_bbox = bbox
                        lendmarks = keypoints

                max_bbox = max_bbox[0:4]

                if frames % 3 == 0:
                    landmarks = np.array([lendmarks['left_eye'][0],lendmarks['right_eye'][0],lendmarks['nose'][0],lendmarks['mouth_left'][0],lendmarks['mouth_right'][0],
                                          lendmarks['left_eye'][1],lendmarks['right_eye'][1],lendmarks['nose'][1],lendmarks['mouth_left'][1],lendmarks['mouth_right'][1]])
                    
                    landmarks = landmarks.reshape((2,5)).T
                    nimg = face_preprocess.preprocess(frame, bbox, landmarks, image_size='112,112')

                    cv2.imwrite(self.args['output'] + '/' + datastring + '.jpg', nimg)
                    cv2.rectangle(frame, (max_bbox[0], max_bbox[1]), (max_bbox[2], max_bbox[3]), (0, 255, 0), 2)
                    cv2.circle(frame,(keypoints['left_eye']), 2, (0,155,255), 2)
                    cv2.circle(frame,(keypoints['right_eye']), 2, (0,155,255), 2)
                    cv2.circle(frame,(keypoints['nose']), 2, (0,155,255), 2)
                    cv2.circle(frame,(keypoints['mouth_left']), 2, (0,155,255), 2)
                    cv2.circle(frame,(keypoints['mouth_right']), 2, (0,155,255), 2)
                    print(f"Faces: {faces} Frames: {frames} ")
                    faces += 1
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"Total faces: {faces} Total frames: {frames} ")
        return faces,frames
    

if __name__ == "__main__":
    args = {'output':'../data/yash_baravaliya','faces':50}
    dt = DataTraining(args)
    dt.get_face()




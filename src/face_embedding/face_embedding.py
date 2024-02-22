import sys
from src.insightface.deploy import face_model
from imutils import paths
import numpy as np
import pickle
import cv2
import os

sys.path.append('../insightface/deploy')
sys.path.append('../insightface/src/common')

class GenrateFaceEmbedding:

    def __init__(self, args):
        self.args = args
        self.image_size = '112,112'
        self.model = "../insightface/models/model-y1-test2/model,0"
        self.threshold = 1.24
        self.det = 0

    def get_embedding(self):
        print("[INFO] loading face detector...")
        imagePaths = list(paths.list_images(self.args['dataset']))

        # load the face embedding model
        embedding_model = face_model.FaceModel(self.image_size, self.model, self.threshold, self.det)

        # initialize our lists of extracted facial embeddings and corresponding people names
        knownEmbeddings = []
        knownNames = []

        # initialize the total number of faces processed
        total = 0

        # loop over the image paths
        for (i, imagePath) in enumerate(imagePaths):
            # Extract student name from the image path
            print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
            name = imagePath.split(os.path.sep)[-2]

            # load image
            image = cv2.imread(imagePath)
            nimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            nimg = np.transpose(nimg, (2,0,1))

            # get the face embedding for the face in the image
            face_embedding = embedding_model.get_feature(nimg)

            # add the name of the person + corresponding face embedding to their respective lists
            knownNames.append(name)
            knownEmbeddings.append(face_embedding)
            total += 1

        # save the facial embeddings + names to disk
        print("[INFO] serializing {} encodings...".format(total))
        # save to output
        data = {"embeddings": knownEmbeddings, "names": knownNames}
        f = open(self.args.embeddings, "wb")
        f.write(pickle.dumps(data))
        f.close()


if __name__ == "__main__":
    args = {
        "dataset": "../data",
        "embeddings": "../output/embeddings.pickle"
    }
    embedding = GenrateFaceEmbedding(args)
    embedding.get_embedding()
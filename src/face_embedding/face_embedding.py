import sys
from src.insightface.deploy import face_model
from imutils import paths
import numpy as np
import pickle
import cv2
import os

sys.path.append('../insightface/deploy')
sys.path.append('../insightface/src/common')


class GenerateFaceEmbedding:

    def __init__(self, args):
        self.args = args
        self.image_size = '112,112'
        self.model = "../insightface/models/model-y1-test2/model,0"
        self.threshold = 1.24
        self.det = 0

    def genFaceEmbedding(self):
        # Grab the paths to the input images in our dataset
        print("[INFO] quantifying faces...")
        imagePaths = list(paths.list_images(self.args['dataset']))

        # Initialize the faces embedder
        embedding_model = face_model.FaceModel(self.image_size, self.model, self.threshold, self.det)

        # Initialize our lists of extracted facial embeddings and corresponding people names
        knownEmbeddings = []
        knownNames = []

        # Initialize the total number of faces processed
        total = 0

        # Loop over the imagePaths
        for (i, imagePath) in enumerate(imagePaths):
            # extract the person name from the image path
            print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
            name = imagePath.split(os.path.sep)[-2]

            # load the image
            image = cv2.imread(imagePath)
            # convert face to RGB color
            nimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            nimg = np.transpose(nimg, (2, 0, 1))
            # Get the face embedding vector
            face_embedding = embedding_model.get_feature(nimg)

            # add the name of the person + corresponding face
            # embedding to their respective list
            knownNames.append(name)
            knownEmbeddings.append(face_embedding)
            total += 1

        print(total, " faces embedded")

        # save to output
        output_dir = os.path.dirname(self.args['embeddings'])
        os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist
        data = {"embeddings": knownEmbeddings, "names": knownNames}
        with open(self.args['embeddings'], "wb") as f:
            pickle.dump(data, f)

# ...

if __name__ == "__main__":
    args = {
        "dataset": "../data",
        "embeddings": "../Models/embeddings.pickle"
    }
    genEmbedding = GenerateFaceEmbedding(args)
    genEmbedding.genFaceEmbedding()

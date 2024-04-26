import os
import pickle
import cv2
from insightface.app import FaceAnalysis

# Load the buffalo_s model
face_analysis = FaceAnalysis(name='buffalo_s')
face_analysis.prepare(ctx_id=0, det_size=(256, 256)) # ctx_id is used for computation type (0 = CPU & 1 = GPU)

base_folder = "data" # Path of Dataset
output_pickle_file = "face_Embedding.pickle"

embeddings_dict = {} # Empty Dict

for subfolder in os.listdir(base_folder):
    subfolder_path = os.path.join(base_folder, subfolder)

    # Skip files (only process Subfolders)
    if not os.path.isdir(subfolder_path):
        continue

    embeddings = []

    # Iterate through each image in the subfolder
    for image_name in os.listdir(subfolder_path):
        image_path = os.path.join(subfolder_path, image_name)
        img = cv2.imread(image_path)

        # Perform face embedding
        embedding = face_analysis.get(img)
        embeddings.append(embedding[0]['embedding'])  # Extract and append only the embedding value

    # Store the embeddings for this Subfolder (class)
    embeddings_dict[subfolder] = embeddings

    print("Embedding of", subfolder," is successful.")
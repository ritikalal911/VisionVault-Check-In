import os
import pickle
import cv2
from insightface.app import FaceAnalysis

def generate_embeddings(base_folder, output_pickle_file, success_callback=None, ctx_id=1, det_size=(256, 256), model_name='buffalo_s'):
    """
    Generate embeddings for faces in images found in subfolders of the specified base folder.

    Args:
    - base_folder (str): Path of the dataset where each subfolder contains images of a single person.
    - output_pickle_file (str): Path of the output pickle file where the embeddings will be saved.
    - success_callback (function): Callback function to be called when an embedding is successfully generated.
    - ctx_id (int): Context ID used for computation type (0 for CPU, 1 for GPU).
    - det_size (tuple): Size of the detector.
    - model_name (str): Name of the insightface model to use.

    Returns:
    None
    """

    # Load the face analysis model
    face_analysis = FaceAnalysis(name=model_name)
    face_analysis.prepare(ctx_id=ctx_id, det_size=det_size)

    embeddings_dict = {}  # Empty dictionary to store embeddings

    # Iterate through subfolders in the base folder
    for subfolder in os.listdir(base_folder):
        subfolder_path = os.path.join(base_folder, subfolder)

        # Skip files (only process subfolders)
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

        # Store the embeddings for this subfolder (class)
        embeddings_dict[subfolder] = embeddings

        # Call the success callback function if provided
        if success_callback is not None:
            success_callback(subfolder)

    # Save the embeddings dictionary as a pickle file
    with open(output_pickle_file, 'wb') as handle:
        pickle.dump(embeddings_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Embeddings saved to:", output_pickle_file)

if __name__ == "__main__":
    base_folder = "data"  # Path of Dataset
    output_pickle_file = os.path.join("src", "Models", "embeddings.pickle")  # Output pickle file path

    # Define a success callback function to be called when an embedding is successfully generated
    def success_callback(subfolder):
        status = f"Embedding of {subfolder} is successful."
        print(status)

    # Call the generate_embeddings function with the success callback
    generate_embeddings(base_folder, output_pickle_file, success_callback)

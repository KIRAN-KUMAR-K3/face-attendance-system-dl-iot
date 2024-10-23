import os
import cv2
import numpy as np
from PIL import Image

def train():
    # Create the LBPH face recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # Load the face detector
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    path = 'dataSet'

    def getImagesWithID(path):
        # Get a list of image paths in the dataset folder
        imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.jpg', '.png'))]

        faces = []
        IDs = []
        
        for imagePath in imagePaths:
            # Open and convert image to grayscale
            faceImg = Image.open(imagePath).convert('L')
            faceNp = np.array(faceImg, 'uint8')
            # Extract ID from the filename
            try:
                ID = int(os.path.split(imagePath)[-1].split('.')[1])  # e.g., "user.1.jpg" -> ID = 1
            except ValueError:
                print(f"Skipping image {imagePath} due to invalid ID format.")
                continue
            
            faces.append(faceNp)
            IDs.append(ID)
            print(f"Training on image: {imagePath}, ID: {ID}")  # Print processed image and ID
            cv2.imshow("training", faceNp)
            cv2.waitKey(10)

        return np.array(IDs), faces

    # Get images and IDs
    Ids, faces = getImagesWithID(path)
    
    if len(Ids) == 0:
        print("No valid images found for training.")
        return

    # Train the recognizer
    recognizer.train(faces, Ids)
    # Save the trained data
    recognizer.save('recognizer/trainingData.yml')
    print("Training complete and data saved to 'recognizer/trainingData.yml'.")
    cv2.destroyAllWindows()

train()


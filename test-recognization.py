import cv2
import numpy as np

# Load the trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('recognizer/trainingData.yml')

# Load the face cascade classifier
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Create a dictionary to map IDs to names
id_to_name = {
    123: "KIRAN KUMAR K",  # Change these to match your dataset IDs and names
    1: "Person2",
    2: "Person3",
    # Add more mappings as needed
}

def recognize_faces():
    # Start capturing from the webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Loop through the detected faces
        for (x, y, w, h) in faces:
            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Predict the ID of the detected face
            face_id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            
            # Check if the confidence is low enough (higher accuracy)
            if confidence < 100:
                name = id_to_name.get(face_id, "Unknown")
                confidence_text = f"{round(100 - confidence)}%"
            else:
                name = "Unknown"
                confidence_text = f"{round(100 - confidence)}%"

            # Display the name and confidence on the frame
            cv2.putText(frame, f"{name} - {confidence_text}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Show the frame
        cv2.imshow("Face Recognition", frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces()

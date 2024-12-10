import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Set image dimensions
image_size = (224, 224)

# Load dataset (example: 'dataset' folder with subfolders for each person)
def load_images_from_folder(folder):
    images = []
    labels = []
    label = 0
    for person_name in os.listdir(folder):
        person_folder = os.path.join(folder, person_name)
        if os.path.isdir(person_folder):
            for image_name in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_name)
                image = load_img(image_path, target_size=image_size)
                image = img_to_array(image)
                image = image / 255.0  # Normalize image
                images.append(image)
                labels.append(label)
            label += 1
    return np.array(images), np.array(labels)

# Path to your dataset (folder structure: 'dataset/person_name/image.jpg')
dataset_path = 'dataset'  # Adjust the path to your dataset

# Load and preprocess images and labels
images, labels = load_images_from_folder(dataset_path)
labels = to_categorical(labels)  # One-hot encode labels

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define the CNN model
model = Sequential()

# Add Convolutional and MaxPooling layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output and add fully connected layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(os.listdir(dataset_path)), activation='softmax'))  # Number of classes = number of people

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Summarize the model architecture
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model as .h5
model.save('face_recognition_model.h5')
print("Model saved as 'face_recognition_model.h5'")

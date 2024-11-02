# Face Recognition Based Attendance System Using Deep Learning and IoT

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This project implements a Face Recognition Based Attendance System using Deep Learning techniques integrated with IoT. The system automatically detects and recognizes faces of students in real-time, marking attendance accordingly. This project aims to streamline the attendance process in educational institutions and minimize the potential for proxy attendance.

## Features
- Real-time face recognition for attendance marking
- Support for multiple users with string-based IDs (e.g., "4AL22CS405")
- Efficient and accurate detection using deep learning algorithms
- IoT integration for remote monitoring and attendance tracking
- User-friendly interface for viewing attendance reports

## Technologies Used
- Python
- OpenCV
- TensorFlow / Keras (for deep learning)
- Flask (for web interface)
- Raspberry Pi / Arduino (for IoT integration)
- Haar Cascade Classifier (for face detection)

## Installation
To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/KIRAN-KUMAR-K3/face-attendance-system-dl-iot.git
   cd face-attendance-system-dl-iot
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the Haar Cascade file: Ensure the `haarcascade_frontalface_default.xml` file is in the project directory.

## Usage
1. Run the dataset creator to gather training data:
   ```bash
   python datasetCreator.py
   ```

2. Train the model using the collected data:
   ```bash
   python trainner.py
   ```

3. Start the application:
   ```bash
   python app.py
   ```

4. Access the web interface: Open your web browser and navigate to [http://localhost:5000](http://localhost:5000) (or your device's IP address if using IoT).

5. Mark attendance: The system will automatically detect and recognize faces as they come into the camera's view.

## Directory Structure
```
face-attendance-system-dl-iot/
├── dataSet/                             # Directory for storing training images
├── datasetCreator.py                    # Script to collect training data
├── haarcascade_frontalface_default.xml  # Pre-trained face detection model
├── trainner.py                          # Script to train the model
├── app.py                               # Main application script
└── requirements.txt                     # Python dependencies
```

## Contributing
Contributions are welcome! If you have suggestions for improvements or want to report issues, please create an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

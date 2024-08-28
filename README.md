# Parking Space Detection using Flask and OpenCV

This project is a web application that detects and counts free and occupied parking spaces in a video feed. It uses a pre-trained Convolutional Neural Network (CNN) model to classify whether a parking space is occupied by a car or not. The application is built using Flask, OpenCV, and TensorFlow/Keras.

## Features

- **Real-time Video Processing:** Detects cars in real-time from a video feed.
- **Parking Space Detection:** Classifies each predefined parking space as "car" or "no car".
- **Live Video Stream:** Displays the video feed with overlaid parking space information.
- **API for Space Count:** Provides an API endpoint to get the current count of free and occupied spaces.

## Technologies Used

- **Flask:** A micro web framework for Python.
- **OpenCV:** A library for real-time computer vision.
- **TensorFlow/Keras:** A deep learning framework used for training and running the CNN model.
- **NumPy:** A library for numerical computations in Python.
- **Pickle:** Used to load the saved parking space positions.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Chando0185/car_parking_detection_space_count.git
cd parking-space-detection
```

### 2. Install Dependencies

Make sure you have Python installed. Then, install the required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Prepare the Model and Data

- **Model:** Ensure the trained model (`model_final.h5`) is placed in the project directory.
- **Parking Positions:** The parking space positions are stored in `carposition.pkl`. Ensure this file is also in the project directory.
- **Video Feed:** Place your test video (`car_test.mp4`) in the project directory.

### 4. Run the Application

Start the Flask application:

```bash
python app.py
```

The application will be available at `http://127.0.0.1:5000/`.

### 5. Access the Application

- **Homepage:** View the video stream at `http://127.0.0.1:5000/`.
- **Video Feed:** The video feed can be accessed at `http://127.0.0.1:5000/video_feed`.
- **Space Count API:** Get the current free and occupied space count via `http://127.0.0.1:5000/space_count`.

## Project Structure

- `app.py`: The main Flask application file.
- `model_final.h5`: The pre-trained CNN model used for detecting cars in parking spaces.
- `carposition.pkl`: Pickled list of parking space positions.
- `car_test.mp4`: The test video file.
- `templates/index.html`: The HTML template for the web interface.
- `requirements.txt`: A list of Python dependencies.

## How It Works

1. **Video Processing:** The application reads frames from the video feed.
2. **Parking Space Detection:** Each frame is processed to extract regions corresponding to predefined parking spaces. The CNN model classifies these regions as either "car" or "no car".
3. **Overlay Information:** The application overlays the classification results on the video frame and counts the number of free and occupied spaces.
4. **API Responses:** The `/space_count` endpoint provides real-time data about parking space availability.

## Dependencies

- Flask
- OpenCV
- TensorFlow/Keras
- NumPy
- Pickle

You can install the dependencies using:

```bash
pip install -r requirements.txt
```

## Contributing

Feel free to submit issues, fork the repository, and send pull requests.

---

Feel free to modify the above content to suit your project's specific details and requirements!

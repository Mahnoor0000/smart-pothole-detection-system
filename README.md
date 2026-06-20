# Smart Pothole Detection System

A computer vision-based web application that detects potholes in road images and videos using a YOLO segmentation model. The project is designed to support road-condition assessment by identifying potholes, showing detection confidence, estimating severity, and generating simple inspection reports.

## Features

- Detects potholes in road images
- Supports video-based pothole detection
- Shows detection confidence score
- Counts detected potholes
- Estimates road severity as Low, Medium, or High
- Provides a Streamlit dashboard for easy interaction
- Allows CSV report download for detection results

## Technologies Used

- Python
- YOLO
- OpenCV
- Streamlit
- NumPy
- Pandas
- Pillow
- Ultralytics

## How It Works

The user uploads a road image or video through the Streamlit interface. The YOLO model processes the input and detects potholes using segmentation. The app then displays the detected potholes with visual markings, calculates the number of potholes, shows the average confidence score, and assigns a severity level based on the detection result.

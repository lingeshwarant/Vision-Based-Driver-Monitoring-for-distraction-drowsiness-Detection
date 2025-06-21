# Vision-Based-Driver-Monitoring-for-distraction-drowsiness-Detection

# Drowiness:
![image](https://github.com/lingeshwarant/Vision-Based-Driver-Monitoring-for-distraction-drowsiness-Detection/blob/main/drowsy.png)

# Distraction:
![image](https://github.com/lingeshwarant/Vision-Based-Driver-Monitoring-for-distraction-drowsiness-Detection/blob/main/distracted.png)

# Overview:
Drowsiness and distraction detection system is a project designed to detects and alert the driver in real-time if remains drowsy or distracted from driving. Also seatbelt detection also added as additional feature.

## Features

- **Real-time Monitoring**: (OpenCV) Detects signs of drowsiness and distraction using a webcam or video input.
- **Model Inference**: (ONNX Runtime + YOLOv8) Run deep learning inference on frames to detect driver behaviors.
- **Detection History and Video Threading**: Maintain time-based behavior tracking (yawns, drowsiness, distractions) and process video frames asynchronously without freezing the UI.
- **Light Weight and Edge ready**: Small model size, low memory usage and suits devices like Raspberry Pi, Jetson Nano.
- **Alert System**: (QSound, Conditional Logic) Play audio and display visual alerts when unsafe behavior is detected.
- **User Interface**: Display live video, detection results, and alerts in a user-friendly interface

---


## Key Files

- `Training_Script.ipynb`: Script for Training the labelled dataset using yolov8n in Google colab.
- `Drowsiness_Detector_GUI.py`: Core detection script integrating real-time inference and alerts.
- `model_format_convertion.ipynb`: Convert the .pt file trained weights into .onnx format.
- `Merge_script.py`: Merge the labelled datasets with same folder structure in yolo format.

---

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/tyrerodr/Real_time_drowsy_driving_detection.git
    cd Real_time_drowsy_driving_detection
    ```

2. **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the detection system:**
    ```bash
    python Drowsiness_Detector_GUI.py
    ```

---


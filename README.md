# Vision-Based-Driver-Monitoring-for-distraction-drowsiness-Detection

# Drowiness:
![image](https://github.com/lingeshwarant/Vision-Based-Driver-Monitoring-for-distraction-drowsiness-Detection/blob/main/Output_images/drowsy.png)

# Distraction:
![image](https://github.com/lingeshwarant/Vision-Based-Driver-Monitoring-for-distraction-drowsiness-Detection/blob/main/Output_images/distracted.png)

# Overview:
Drowsiness and distraction detection system is a project designed to detects and alerts the drivers in real-time if remains drowsy or distracted from driving. Also seatbelt detection is added as additional feature.

## Features

- **Real-time Monitoring**: Detects signs of drowsiness and distraction using a webcam or video input.
- **Indirect Phone usage detection**: Detects the phone usage of the driver even though the phone is not visible currently (if detected previously).
- **Detection History and Video Threading**: Maintain time-based behavior tracking (yawns, drowsiness, distractions) and process video frames asynchronously without freezing the UI.
- **Light Weight and Edge ready**: Small model size, low memory usage and suits devices like Raspberry Pi, Jetson Nano.
- **False positive handling**: If a strong drowsiness or yawning condition exists, the system prioritizes those alerts over indirect phone usage to avoid confusion.
- **User Interface**: Display live video, detection results, and alerts in a user-friendly interface

---


## Key Files

- `Training_Script.ipynb`: Script for Training the labelled dataset using yolov8n in Google colab.
- `Drowsiness_Detector_GUI.py`: Core detection script integrating real-time inference and alerts.
- `model_conversion.ipynb`: Convert the .pt file trained weights into .onnx format.
- `Merge_script.py`: Merge the labelled datasets with same folder structure in yolo format.

---

## Model Architecture – YOLOv8n
- Project uses YOLOv8n (nano) models from Ultralytics, which are optimized for real-time object detection.
- `Input`: 640 × 640 × 3 image - RGB image, resized & normalized
- `Backbone`: C2f layers (Conv + BN + SiLU) - Extracts deep features from the image
- `Neck`: FPN + PAN (Feature Pyramid & Path Aggregation) - Combines features at multiple scales
- `Head`: Detect layer with anchor-free design - Predicts bounding boxes, class scores, objectness
- `Output`: Array of detections → [x1, y1, x2, y2, confidence, class_id] - For every object detected

---


## How It Works

The system uses two separate YOLOv8 models:

1. **Drowsiness Model:**
   - Classifies in Classes: `drowsy’, ‘awake’, ‘yawn’, ‘distraction’, ‘head drop’ ‘Phone’ , ‘smoking.
   - Trained on public and collected datasets:
     - [Drowsiness collected](https://universe.roboflow.com/addicons/drowsiness-sgvf2-fd3v9/dataset/2)
     - [Drowsiness_public](https://universe.roboflow.com/saujanya-shankar/drowsiness-sgvf2)
   - Merging the two datasets can be done using the script "Merge_script.py"

2. **Seatbelt Model:**
   - Detects Seatbelt (Driver worn seatbelt) vs NoSeatbelt (seatbelt not worn by driver).
   - Trained on:
     - [Seatbelt Dataset](https://universe.roboflow.com/object-detection-0t04j/seatbelt-j3w5q)

Once trained, the models' predictions are combined with confidence thresholds and visualized in a PyQt5 GUI.

---

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/lingeshwarant/Vision-Based-Driver-Monitoring-for-distraction-drowsiness-Detection.git
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


## Technologies Used

- **Python**
- **YOLOv8** – Object detection framework.
- **OpenCV** – Computer vision tasks.
- **ONNX Runtime** – Model Inference.
- **Pytorch** – Model training.
- **PyQt5** – Graphical user interface.

---


## Future Improvements

- **Integration with Wearables:** Add heart rate or other vitals monitoring.
- **Design GRU-Based Classifier:** Use GRU to learn temporal patterns - 1 or 2 GRU layers and Final classification via dense layer.
- **Noise Tolerance & Robustness:** Augment training with Frame skips, Blur, low-light simulation and Occlusions.




**Lingeshwaran T**  
Associate Software Engineer  
Bosch Global Software Technologies, Coimbatore 
Email: pmlingesh123@gmail.com 

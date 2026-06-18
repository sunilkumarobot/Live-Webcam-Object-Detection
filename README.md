# Live Webcam Object Detection

## Project Overview

This project demonstrates real-time object detection using the YOLOv4-Tiny deep learning model and OpenCV's DNN module.

The system captures frames from a webcam, processes each frame using YOLOv4-Tiny, and detects objects in real time. Detected objects are highlighted with bounding boxes, class labels, and confidence scores.

YOLO (You Only Look Once) is a single-stage object detection algorithm that performs object localization and classification in a single forward pass, making it suitable for real-time computer vision applications.

---

## Features

* Real-time webcam object detection
* YOLOv4-Tiny implementation using OpenCV
* Bounding box visualization
* Confidence score display
* Non-Maximum Suppression (NMS)
* Live video processing
* Fast inference speed

---

## Detection Pipeline

```text
Webcam
   ↓
Capture Frame
   ↓
Resize (416×416)
   ↓
Blob Creation
   ↓
YOLOv4-Tiny Network
   ↓
Objectness Check
   ↓
Confidence Check
   ↓
Non-Maximum Suppression
   ↓
Bounding Boxes
   ↓
Display Output
```

---

## Model Files

The project uses:

```text
coco.names
yolov4-tiny.cfg
yolov4-tiny.weights
```

### coco.names

Contains object class labels such as:

* Person
* Chair
* Bottle
* Laptop
* Cell Phone
* Backpack
* Car
* Dog

and many more.

---

## Important Concepts

### Objectness Score

Objectness score represents the probability that a bounding box contains an object.

Example:

```text
Objectness = 0.95
```

Meaning:

```text
95% confidence that an object exists inside the predicted box.
```

---

### Confidence Score

Confidence score represents how confident the model is about the detected class.

Example:

```text
Person = 0.93
Chair = 0.04
Bottle = 0.02
```

The object is classified as:

```text
Person
```

because it has the highest confidence score.

---

### Non-Maximum Suppression (NMS)

NMS removes duplicate overlapping bounding boxes and keeps only the highest confidence prediction.

Example:

Before NMS:

```text
Car Box 1 = 95%
Car Box 2 = 90%
Car Box 3 = 87%
```

After NMS:

```text
Keep Box 1
Remove Box 2
Remove Box 3
```

---

## Installation

Install required libraries:

```bash
pip install -r requirements.txt
```

---

## Run the Project

```bash
python yolo4.py
```

The webcam will start automatically and begin detecting objects.

Press:

```text
ESC
```

to exit the application.

---

## Demo Video

A live webcam demonstration of the project is available below:

```markdown
[Watch Demo Video](videos/webcam_demo.mp4)
```

If you upload the video to YouTube, replace it with:

```markdown
## Demo Video

Watch the project demonstration here:

https://youtu.be/YOUR_VIDEO_LINK
```

---

## Output

The system successfully performs real-time object detection and can detect common COCO dataset objects such as:

* Person
* Chair
* Bottle
* Cup
* Laptop
* Keyboard
* Mouse
* Cell Phone
* Backpack

depending on the objects visible in front of the webcam.

---

## Learning Outcomes

Through this project, I gained hands-on experience in:

* Deep Learning Based Object Detection
* YOLOv4-Tiny Architecture
* OpenCV DNN Module
* Blob Creation
* Forward Propagation
* Confidence Thresholding
* Objectness Thresholding
* Non-Maximum Suppression
* Real-Time Video Processing
* Webcam Integration

---

## Future Improvements

* YOLOv8 Implementation
* Object Tracking using DeepSORT
* ROS2 Integration
* Autonomous Robot Vision Systems
* Multi-Camera Detection

---

## Author

**Sunil Kumar**

Robotics Engineer | Computer Vision Enthusiast | OpenCV Developer

---

### Folder Structure

```text
YOLOv4-Tiny-Live-Object-Detection
│
├── images/
│
├── videos/
│   └── webcam_demo.mp4
│
├── models/
│   ├── coco.names
│   └── yolov4-tiny.cfg
│
├── yolo4.py
├── requirements.txt
└── README.md
```

<img width="496" height="402" alt="image" src="https://github.com/user-attachments/assets/ee051d81-511d-4be9-97d0-d95ba3f92eba" />











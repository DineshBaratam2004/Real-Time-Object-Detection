```md
# 🚀 Real-Time Object Detection using YOLOv8

This project implements a **Real-Time Object Detection system** using **YOLOv8**. It supports training a custom dataset and performing real-time detection using a webcam or video feed. The dataset is prepared and managed using **Roboflow**, and the model is trained using the **Ultralytics YOLOv8 framework**.

---

## 📁 Project Files Overview

```

Real-Time-Object-Detection/
│
├── README.md                 # Main project documentation
├── README.dataset.txt        # Dataset details
├── README.roboflow.txt       # Roboflow dataset information
├── data.yaml                 # Dataset configuration file
├── train_yolo.py             # YOLOv8 training script
├── final_detect.py           # Real-time object detection script
├── yolov8s.pt                # Pre-trained YOLOv8 model weights

````

---

## ✨ Features

- 🎥 Real-time object detection
- 🧠 YOLOv8 deep learning model
- 🏷️ Bounding boxes with class labels
- ⚡ Fast and accurate detection
- 🔁 Custom dataset training support
- 📦 Roboflow-integrated dataset handling

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Framework:** Ultralytics YOLOv8  
- **Libraries:**  
  - OpenCV  
  - PyTorch  
  - NumPy  
- **Dataset Tool:** Roboflow  

---

## ⚙️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/real-time-object-detection.git
````

2. **Navigate to the project directory**

```bash
cd real-time-object-detection
```

3. **Install required dependencies**

```bash
pip install ultralytics opencv-python torch numpy
```

---

## 📊 Dataset Configuration

* Dataset details are provided in:

  * `README.dataset.txt`
  * `README.roboflow.txt`
* Dataset paths and class names are defined in `data.yaml`

---

## 🧠 Model Training

To train the YOLOv8 model on your custom dataset:

```bash
python train_yolo.py
```

* Uses `data.yaml` for dataset configuration
* Uses `yolov8s.pt` as the base pre-trained model

---

## ▶️ Real-Time Detection

To run real-time object detection using webcam:

```bash
python final_detect.py
```

* Press **`Q`** to exit the detection window
* Displays object name and confidence score

---

## 📈 Output

* Live webcam feed
* Bounding boxes around detected objects
* Class labels with confidence percentages

---

## 🚧 Challenges Faced

* Handling real-time inference speed
* Dataset annotation accuracy
* Optimizing detection performance
* Managing lighting and background variations

---

## 🔮 Future Improvements

* 🎯 Increase detection accuracy
* 📱 Deploy as a web or mobile application
* 🧠 Add object tracking and counting
* ☁️ Cloud-based inference
* 🔄 Model optimization for higher FPS

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 🙌 Acknowledgements

* Ultralytics YOLOv8
* Roboflow
* OpenCV Community
* PyTorch Team

```

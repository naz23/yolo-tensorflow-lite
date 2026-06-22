# YOLO Object Detection with TensorFlow Lite (TFLite)

This repository contains a Python script to run YOLO models (including YOLOv8 and YOLO26) using TensorFlow Lite (`tflite-runtime`) with OpenCV for real-time webcam detection.

The script automatically detects the output shape of the loaded model and determines if it is YOLOv8-style (raw grid output) or YOLO26-style (pre-NMS detections).

## Features
- Real-time object detection via webcam (OpenCV).
- Automatic format detection for YOLOv8 and YOLOv10+/YOLO26.
- Supports both float16 and float32 models.
- Standard Non-Maximum Suppression (NMS) for raw grid output models.
- Displays FPS (Frames Per Second), Average FPS, and the number of active detections.

## Prerequisites
Before running the script, make sure you have Python installed (3.8+ recommended).

### Install Dependencies
You can install the required packages using pip. It's recommended to do this within a virtual environment.

```bash
# Create a virtual environment (optional)
python -m venv venv
# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

> [!NOTE]
> If you have trouble installing `tflite-runtime` directly on your platform, you can also install the full `tensorflow` library (which includes the TFLite interpreter) and adjust the import statement in `main.py` from `import tflite_runtime.interpreter as tflite` to `from tensorflow.lite.python.interpreter import Interpreter` (or similar).

## How to Run the Program

1. Ensure your webcam is connected.
2. Select your desired YOLO TFLite model in `main.py` (line 9):
   ```python
   model_path = "yolo26n_float32.tflite"  # or "yolov8n_float32.tflite", etc.
   ```
3. Run the python script:
   ```bash
   python main.py
   ```
4. A window titled **"YOLO TFLite Detection"** will open showing your webcam feed with detection boxes.
5. Press the **`ESC`** key to exit the program.

## Performance & Benchmarks
Tested on a **Raspberry Pi 4 Model B** (using 4 threads) with a **320x320 resolution**:
- **YOLOv8n**: 3 ~ 5 FPS
- **YOLO26n**: 4 ~ 6 FPS

## Available Models in this Repository
- `yolo26n_float16.tflite`
- `yolo26n_float32.tflite`
- `yolov8n_float16.tflite`
- `yolov8n_float32.tflite`
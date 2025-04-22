# Object Detection for self Driving Vehicles with YOLO

This project implements real-time object detection using the YOLO (You Only Look Once) model. YOLO is a state-of-the-art object detection system that can detect multiple objects in images and videos with high accuracy and speed.

## Project Overview

The system uses YOLOv5, which is known for its:
- Real-time object detection capabilities
- High accuracy in detecting multiple objects simultaneously
- Efficient performance on both CPU and GPU
- Support for custom model training

## Data Pipeline

The project implements a robust data pipeline specifically designed for self-driving vehicle scenarios:

### 1. Data Input Sources
- Vehicle-mounted cameras (front, side, rear views)
- Video streams (MP4, AVI formats)
- Image sequences (JPG, PNG)
- Real-time camera feeds

### 2. Preprocessing Pipeline
```
Raw Input → Resizing → Normalization → Color Space Conversion → Augmentation
```
- **Resizing**: Standardizes images to 640x640 pixels
- **Normalization**: Scales pixel values to [0,1] range
- **Color Space**: Converts to RGB format
- **Augmentation**: 
  - Random brightness/contrast
  - Motion blur simulation
  - Weather condition simulation (rain, snow, fog)
  - Time-of-day variations

### 3. Detection Pipeline
```
Preprocessed Image → YOLO Model → Raw Detections → Post-processing → Final Output
```
- **Object Classes**: 
  - Vehicles (cars, trucks, buses)
  - Pedestrians
  - Traffic signs
  - Traffic lights
  - Road markings
  - Obstacles

### 4. Post-processing
- Non-Maximum Suppression (NMS)
- Confidence threshold filtering
- Temporal smoothing for video
- Object tracking across frames
- Distance estimation

### 5. Output Formats
- Annotated images/video
- JSON detection results
- CSV logs with:
  - Object coordinates
  - Confidence scores
  - Object classes
  - Frame timestamps
  - Distance estimates

### 6. Performance Metrics
- Average Precision (AP)
- Mean Average Precision (mAP)
- Intersection over Union (IoU)
- Frame processing speed (FPS)
- Detection latency

## Features

- Real-time object detection in images and video streams
- Support for multiple object classes
- Confidence score display for detections
- Bounding box visualization
- Batch processing capability for multiple images
- Easy-to-use interface for both beginners and advanced users

## Prerequisites

- Python 3.8 or higher
- CUDA toolkit (optional, for GPU acceleration)

## Installation

1. Clone the repository:
```bash
git clone this_repository
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
object-detection/
│
├── data.yaml           # Class configurations
├── models/             # Directory for model files
│   └── yolov5m.pt     # Pre-trained YOLO model
├── data/              # Data directory
│   ├── images/        # Input images
│   └── output/        # Detection results
└── requirements.txt   # Project dependencies
```

## Usage

### Basic Usage

The main script `use_model.py` provides several functions for object detection:

```python
from predict import load_model, predict_and_display

# Load the model
model = load_model('models/yolov5m.pt')

# Single image detection
predict_and_display(model, 'data/images/example.jpg')

# Batch processing
images = ['image1.jpg', 'image2.jpg', 'image3.jpg']
for image in images:
    predict_and_display(model, f'data/images/{image}')
```

### Command Line Interface

Run directly from command line:

```bash
python predict.py --source path/to/image --model models/yolov5m.pt --conf 0.25
```

Arguments:
- `--source`: Path to input image/video/directory
- `--model`: Path to YOLO model file
- `--conf`: Confidence threshold (0-1)
- `--save-txt`: Save results to text file
- `--view-img`: Display results

## Core Functions

### Model Operations
- `load_model(model_path)`: Loads and initializes the YOLO model
- `load_classes(yaml_path)`: Loads class names from YAML configuration
- `predict_and_display(model, image_path)`: Performs detection and visualization

### Utility Functions
- Image preprocessing
- Bounding box drawing
- Confidence score calculation
- Result saving and export

## Performance Optimization

The system includes several optimization features:
- Batch processing for multiple images
- GPU acceleration support
- Adjustable inference parameters
- Memory management for large datasets

## Error Handling

The system includes comprehensive error handling for:
- Model loading failures
- Invalid input files
- Configuration errors
- Resource unavailability
- GPU memory issues

Common error messages and their solutions are provided in the error output.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv5 by Ultralytics
- PyTorch community
- Open-source computer vision contributors




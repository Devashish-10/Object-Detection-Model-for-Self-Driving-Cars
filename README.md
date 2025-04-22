# Object Detection for Self Driving Vehicles with YOLO

This project implements object detection using the YOLO (You Only Look Once) model, specifically designed for detecting objects in images.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have the following files in your project directory:
   - Your YOLO model file (e.g., `yolov5m_self_driving_car.pt`)
   - The `data.yaml` file containing class names
   - The images you want to process

## Usage

The main script `predict.py` provides functions for loading the model and making predictions:

```python
from predict import load_model, predict_and_display

# Load the model
model = load_model('path/to/your/model.pt')

# Make predictions on an image
predict_and_display(model, 'path/to/your/image.jpg')
```

You can also run the script directly:

```bash
python predict.py
```

This will use the default model and test image paths specified in the main function.

## Functions

- `load_model(model_path)`: Loads a YOLO model from the specified path
- `load_classes(yaml_path)`: Loads class names from a YAML file
- `predict_and_display(model, image_path)`: Makes predictions on an image and displays the results

## Error Handling

The script includes error handling for common issues such as:
- Missing model files
- Invalid image files
- YAML file parsing errors

If you encounter any errors, check the error message for details on how to resolve the issue.


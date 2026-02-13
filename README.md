# Face Recognition System

A comprehensive face recognition system built with OpenCV and Python, featuring multiple enrollment and recognition scripts with advanced augmentation techniques.

## Features

- **Multiple Face Detection Methods**: DNN-based detection (preferred) with Haar Cascade fallback
- **Image Augmentation**: Automatic generation of training variations (rotation, flip, brightness, blur)
- **Real-time Recognition**: Live webcam-based face recognition
- **Multi-person Support**: Enroll and recognize multiple individuals
- **Persistent Storage**: JSON-based label mapping and trained model persistence
- **Flexible Confidence Thresholds**: Adjustable recognition sensitivity

## Project Structure

```
├── images/                          # Captured face images for training
├── models/                          # DNN model files (optional)
├── opencv-files/                    # Additional cascade files
├── training-data/                   # Training dataset folders
├── test-data/                       # Test images
├── cam.py                           # Basic enrollment & recognition
├── FR1.py                           # Advanced system with augmentation
├── FR11.py                          # Multi-person enrollment
├── FR2.py                           # Recognition script (threshold: 85)
├── FR22.py                          # Recognition script (threshold: 50)
├── face_recognizer_model.yml        # Trained LBPH model
├── label_info.json                  # Person ID to name mapping
└── requirements.txt                 # Python dependencies
```

## Installation

### Prerequisites

- Python 3.7 or higher
- Webcam/Camera

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/face-recognition-system.git
cd face-recognition-system
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

> **Note:** The requirements file includes `opencv-contrib-python`.
> Installing `opencv-python` instead will lead to errors like
> ``AttributeError: module 'cv2.face' has no attribute 'LBPHFaceRecognizer_create'``
> during training. If you encounter this, uninstall any existing
> OpenCV packages and reinstall the contrib variant as shown below:
>
> ```bash
> pip uninstall opencv-python opencv-contrib-python
> pip install opencv-contrib-python
> ```

3. (Optional) Download DNN models for improved face detection:
   - Place `deploy.prototxt.txt` and `res10_300x300_ssd_iter_140000.caffemodel` in the `models/` folder
   - Download from: [OpenCV DNN Face Detector](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector)

## Usage

### Option 1: Complete System (Recommended)

Run **FR1.py** for the full-featured system:

```bash
python FR1.py
```

**Menu Options:**
1. Capture images (multi-angle) and Train model
2. Train model from existing images
3. Run recognition (live)
4. Normalize old filenames
5. Exit

**Workflow:**
- Choose option 1 to enroll a new person
- Follow on-screen instructions to capture face images
- Press 'c' to capture or let it auto-capture
- System will automatically augment images and train the model
- Choose option 3 to test recognition
- Press 'e' to exit recognition mode


### Adjustable Parameters (in FR1.py)

```python
CAPTURE_COUNT = 25             # Images per person
CONFIDENCE_THRESHOLD = 110      # Lower = stricter matching
DNN_CONF_THRESHOLD = 0.5        # Face detection confidence
MIN_FACE_SIZE = (60, 60)        # Minimum face dimensions
LBPH_PARAMS = dict(             # LBPH algorithm parameters
    radius=3, 
    neighbors=8, 
    grid_x=10, 
    grid_y=10
)
```

Lower values mean the system requires higher confidence for recognition.

## How It Works

1. **Face Detection**: Uses DNN or Haar Cascade to detect faces in frames
2. **Image Augmentation**: Creates variations (flip, rotate, brightness, crop) to improve training
3. **Training**: LBPH (Local Binary Patterns Histograms) algorithm learns face features
4. **Recognition**: Compares detected faces against trained model
5. **Confidence Scoring**: Returns match confidence (lower = better match)

## Image Augmentation

The system automatically generates 8 variations per captured image:
- Original (resized to 200x200)
- Horizontal flip
- +10° rotation
- -10° rotation
- Slight zoom/crop
- Brightness increase
- Brightness decrease
- Gaussian blur

This improves recognition accuracy across different lighting and angles.

## File Naming Convention

Images are saved as: `{label}_{count}_{variation}_{timestamp}.jpg`

Example: `13_0_2_20260213_230257.jpg`
- Label: 13
- Capture count: 0
- Variation: 2
- Timestamp: 2026-02-13 23:02:57

## Troubleshooting

### Camera Not Opening
- Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` or higher
- Ensure no other application is using the camera

### Low Recognition Accuracy
- Capture more images (60+ recommended)
- Ensure good lighting during capture
- Vary head angles during capture
- Increase `CONFIDENCE_THRESHOLD` value

### "Model not found" Error
- Run training first (FR1.py option 1 or 2)
- Ensure `face_recognizer_model.yml` exists

### Import Errors
- Install opencv-contrib-python (not opencv-python)
- Reinstall: `pip uninstall opencv-python opencv-contrib-python`
- Then: `pip install opencv-contrib-python`

## Dependencies

- **opencv-contrib-python**: Computer vision and face recognition
- **numpy**: Numerical operations
- **imutils**: Image processing utilities

## Technical Details

- **Algorithm**: LBPH (Local Binary Patterns Histograms)
- **Face Detection**: DNN (ResNet-based) or Haar Cascade
- **Image Size**: 200x200 pixels (normalized)
- **Color Space**: Grayscale for processing
- **Storage Format**: YAML for model, JSON for labels

## Future Enhancements

- [ ] Add face verification (1:1 matching)
- [ ] Implement attendance logging
- [ ] Add GUI interface
- [ ] Support for multiple faces in single frame
- [ ] Database integration
- [ ] REST API for remote recognition

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- OpenCV community for excellent documentation
- LBPH algorithm researchers
- Face detection model contributors

## Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: This system is for educational purposes. For production use, consider additional security measures and privacy compliance.

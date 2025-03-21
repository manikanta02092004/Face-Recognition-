# Computer Vision Project: Face Detection and Recognition

## Overview
This project involves face detection and recognition using deep learning techniques. It includes data augmentation, preprocessing, and training a custom neural network model. The project is implemented using Python with OpenCV, NumPy, and PyTorch.

## Project Structure
```
CV_Project_2024-25iiits/
│-- CV project file.ipynb   # Main notebook for dataset processing and face detection
│-- training.ipynb          # Training notebook with model implementation
│-- dataset/                # Contains original dataset
│-- augmented_dataset/      # Augmented dataset for training
│-- .vs/                    # Visual Studio configuration files
│-- archive (7).zip         # Additional data archive (needs extraction)
```

## Dependencies
Ensure you have the following Python libraries installed:
```bash
pip install numpy opencv-python matplotlib torch torchvision mediapipe mtcnn tqdm
```

## Installation Guide
1. Install Python (recommended version: 3.8 or higher).
2. Install required dependencies using the command above.
3. Ensure you have a compatible GPU with CUDA enabled for faster training (optional but recommended).

## Dataset
The dataset is stored in the `dataset/` folder. The `augmented_dataset/` contains variations of images for better training performance.

## How to Run
1. **Dataset Preprocessing:** Run `CV project file.ipynb` to load, visualize, and preprocess face images.
2. **Training the Model:** Execute `training.ipynb` to train the face recognition model. This includes data augmentation and training using PyTorch.
3. **Testing:** After training, test the model with new images to verify its accuracy.

## Usage Examples
To preprocess the dataset and train the model, follow these steps:
```python
# Run the dataset preprocessing script
!python preprocess.py

# Train the model
!python train.py

# Test the model
!python test.py --image test_image.jpg
```

## Model Details
- **Face Detection:** Uses MTCNN for detecting faces.
- **Neural Network:** A custom model (`VGGFace160`) built with PyTorch.
- **Augmentation:** Includes brightness adjustments and noise addition.

## Evaluation Metrics
- **Accuracy:** Measures the percentage of correctly classified faces.
- **Precision & Recall:** Evaluates model effectiveness in recognizing faces.
- **Loss Function:** Uses CrossEntropyLoss for training stability.

## Contribution Guidelines
If you'd like to contribute:
1. Fork the repository.
2. Create a new branch for your feature/fix.
3. Submit a pull request with a detailed explanation.

## License
This project is licensed under the MIT License.

## Notes
- Ensure the dataset is correctly loaded before training.
- Do not modify the `dataset/` folder unless adding new images.
- The project supports Google Colab for cloud-based training.

## Acknowledgments
This project uses open-source libraries and datasets to enhance face recognition capabilities.


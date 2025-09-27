# Pneumonia Detection with Explainable AI

This is a Streamlit web application that uses a deep learning model to detect pneumonia in chest X-ray images. The application provides explainable AI features using Grad-CAM to highlight the regions that influenced the model's decision.

## Features

- **Pneumonia Detection**: Upload chest X-ray images and get predictions for pneumonia presence
- **Explainable AI**: Grad-CAM visualization shows which parts of the image influenced the model's decision
- **User-friendly Interface**: Clean Streamlit interface for easy image upload and result viewing
- **Robust Error Handling**: Comprehensive error handling for missing files and processing issues

## Prerequisites

- Python 3.8 or higher
- A trained PyTorch model file named `pneumonia_classifier.pth` (ResNet-18 based)

## Installation

1. Clone or download this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Make sure your trained model file `pneumonia_classifier.pth` is in the same directory as `streamlit_app.py`
2. Run the Streamlit application:
   ```bash
   streamlit run streamlit_app.py
   ```
3. Open your web browser and navigate to the URL shown in the terminal (usually `http://localhost:8501`)
4. Upload a chest X-ray image using the file uploader
5. View the prediction results and Grad-CAM visualization (if pneumonia is detected)

## Model Requirements

The application expects a PyTorch model with the following characteristics:
- ResNet-18 architecture
- Final layer modified to output 2 classes (NORMAL, PNEUMONIA)
- Trained weights saved as `pneumonia_classifier.pth`

## Dependencies

- streamlit: Web application framework
- torch: PyTorch deep learning framework
- torchvision: Computer vision utilities for PyTorch
- Pillow: Image processing library
- numpy: Numerical computing library
- pytorch-grad-cam: Gradient-based class activation mapping

## Troubleshooting

- **"Model file not found"**: Ensure `pneumonia_classifier.pth` is in the same directory as the application
- **Import errors**: Make sure all dependencies are installed using `pip install -r requirements.txt`
- **Memory issues**: The application runs on CPU by default to ensure compatibility across different systems

## License

This project is for educational and research purposes.
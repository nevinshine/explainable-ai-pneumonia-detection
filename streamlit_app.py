# In streamlit_app.py

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# --- Page Configuration ---
st.set_page_config(
    page_title="Explainable AI for Pneumonia Detection",
    page_icon="🫁",
    layout="centered"
)

# --- Model Loading ---
# Use st.cache_resource to load the model only once, which makes the app faster.
@st.cache_resource
def load_model():
    try:
        # Re-create the model structure (ResNet-18)
        model = models.resnet18()
        num_ftrs = model.fc.in_features
        # Adjust the final layer to have 2 outputs (NORMAL, PNEUMONIA)
        model.fc = nn.Linear(num_ftrs, 2)
        # Load the trained weights from the.pth file
        # Use map_location to ensure the model loads correctly on any machine (CPU or GPU)
        model.load_state_dict(torch.load('pneumonia_classifier.pth', map_location=torch.device('cpu')))
        model.eval() # Set the model to evaluation mode
        return model
    except FileNotFoundError:
        st.error("Model file 'pneumonia_classifier.pth' not found. Please ensure the model file is in the same directory as this app.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# Only load model if we can find the file
try:
    model = load_model()
except:
    st.error("Failed to initialize the model. Please check that all dependencies are installed and the model file exists.")
    st.stop()
# Define the class names in the correct order based on how ImageFolder reads them (alphabetical)
class_names = ['NORMAL', 'PNEUMONIA']

# --- Image Preprocessing ---
def preprocess_image(image_bytes):
    try:
        # Define the transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Open the image from the uploaded bytes
        image = Image.open(image_bytes).convert('RGB')
        # Apply the transformations
        tensor = transform(image).unsqueeze(0) # Add batch dimension
        return image, tensor
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None

# --- Grad-CAM Generation ---
def generate_gradcam(model, input_tensor, original_image):
    try:
        # The target layer is the last convolutional block in ResNet-18
        target_layers = [model.layer4[-1]]
        # We want to explain the 'PNEUMONIA' class, which is at index 1
        targets = [ClassifierOutputTarget(1)]
        
        # Initialize Grad-CAM
        with GradCAM(model=model, target_layers=target_layers) as cam:
            # Generate the heatmap
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :] # Take the first (and only) image in the batch
        
        # Convert the original PIL image to a numpy array for visualization
        rgb_img = np.array(original_image.resize((224, 224)), dtype=np.float32) / 255
        # Overlay the heatmap on the original image
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        return visualization
    except Exception as e:
        st.error(f"Error generating Grad-CAM visualization: {str(e)}")
        return None

# --- Streamlit App Interface ---
st.title("Explainable AI for Pneumonia Detection")
st.write(
    "This application uses a deep learning model to predict whether a "
    "chest X-ray shows signs of pneumonia. It also uses Grad-CAM to create a heatmap, "
    "highlighting the areas the model focused on for its prediction."
)

# File uploader allows user to add their own image
uploaded_file = st.file_uploader("Upload a chest X-ray image...", type=["jpeg", "jpg", "png"])

if uploaded_file is not None:
    # Preprocess the uploaded image
    original_image, input_tensor = preprocess_image(uploaded_file)
    
    if original_image is None or input_tensor is None:
        st.error("Failed to process the uploaded image. Please try a different image.")
        st.stop()
    
    # Use columns for a cleaner layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(original_image, caption='Uploaded X-Ray', use_container_width=True)

    # Get model prediction when the image is uploaded
    try:
        with torch.no_grad():
            outputs = model(input_tensor)
            
            # --- THIS SECTION IS NOW CORRECTED ---
            # Apply softmax to get probabilities across the class dimension (dim=1)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            # Get the top prediction and its confidence
            confidence, pred_idx = torch.max(probabilities, 1)
            prediction = class_names[pred_idx.item()]
            # --- END OF CORRECTION ---

        with col2:
            st.subheader("Prediction")
            st.markdown(f"**Class:** `{prediction}`")
            st.markdown(f"**Confidence:** `{confidence.item():.2%}`")

            # If the model predicts pneumonia, generate and show the explanation
            if prediction == 'PNEUMONIA':
                st.write("Generating explanation heatmap (Grad-CAM)...")
                gradcam_visualization = generate_gradcam(model, input_tensor, original_image)
                if gradcam_visualization is not None:
                    st.image(gradcam_visualization, caption='Model Explanation (Grad-CAM)', use_container_width=True)
                    st.info(
                        "**Heatmap Interpretation:** The red areas highlight the regions of the image that were most influential for the model's decision to classify the image as 'PNEUMONIA'."
                    )
                else:
                    st.warning("Could not generate Grad-CAM visualization.")
            else:
                st.success("The model did not detect signs of pneumonia.")
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.write("Please try uploading a different image or check that the model file is properly loaded.")
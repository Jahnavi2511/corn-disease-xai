import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="Corn Leaf Disease Detection",
    page_icon="🌽",
    layout="wide"
)


# -------------------------------------------------
# CLASS NAMES
# Must be in exactly the same order as training
# -------------------------------------------------

class_names = [
    "Common_rust",
    "Healthy",
    "Leaf_Blight",
    "Leaf_spot",
    "Streak_virus"
]


# -------------------------------------------------
# DEVICE
# -------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# -------------------------------------------------
# IMAGE PREPROCESSING
# Same preprocessing used during training
# -------------------------------------------------

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -------------------------------------------------
# LOAD TRAINED MODEL
# -------------------------------------------------

@st.cache_resource
def load_model():

    model = models.efficientnet_b0(weights=None)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        5
    )

    model.load_state_dict(
        torch.load(
            "models/best_efficientnet_b0.pth",
            map_location=device
        )
    )

    model = model.to(device)

    model.eval()

    return model


model = load_model()


# -------------------------------------------------
# PAGE TITLE
# -------------------------------------------------

st.title("🌽 Corn Leaf Disease Detection")

st.subheader(
    "Cloud-Based Explainable AI System"
)

st.write(
    "Upload a corn leaf image to identify the disease "
    "and visualize the important image regions used by "
    "the AI model."
)

st.divider()


# -------------------------------------------------
# IMAGE UPLOAD
# -------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a Corn Leaf Image",
    type=["jpg", "jpeg", "png"]
)


# -------------------------------------------------
# PREDICTION + GRAD-CAM
# -------------------------------------------------

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    if st.button("🔍 Predict Disease"):

        # -----------------------------------------
        # PREPROCESS IMAGE
        # -----------------------------------------

        input_tensor = eval_transform(
            image
        ).unsqueeze(0).to(device)


        # -----------------------------------------
        # MODEL PREDICTION
        # -----------------------------------------

        with torch.no_grad():

            outputs = model(input_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predicted_class = torch.argmax(
                probabilities,
                dim=1
            ).item()

            confidence = (
                probabilities[0][predicted_class]
                .item() * 100
            )


        # -----------------------------------------
        # GRAD-CAM
        # -----------------------------------------

        target_layers = [
            model.features[-1]
        ]

        cam = GradCAM(
            model=model,
            target_layers=target_layers
        )

        targets = [
            ClassifierOutputTarget(
                predicted_class
            )
        ]

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets
        )[0]


        # -----------------------------------------
        # PREPARE IMAGE FOR OVERLAY
        # -----------------------------------------

        image_np = np.array(
            image.resize((224, 224))
        ).astype(np.float32) / 255.0


        visualization = show_cam_on_image(
            image_np,
            grayscale_cam,
            use_rgb=True
        )


        # -----------------------------------------
        # DISPLAY RESULTS
        # -----------------------------------------

        st.success(
            f"Prediction: "
            f"{class_names[predicted_class]}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.divider()


        col1, col2 = st.columns(2)


        with col1:

            st.subheader(
                "Original Image"
            )

            st.image(
                image,
                use_container_width=True
            )


        with col2:

            st.subheader(
                "Grad-CAM Explanation"
            )

            st.image(
                visualization,
                use_container_width=True
            )


        st.info(
            "The highlighted regions in the Grad-CAM "
            "image indicate the areas that contributed "
            "most strongly to the model's prediction."
        )
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import time

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from lime import lime_image
from skimage.segmentation import mark_boundaries

# ==============================================================================
# MODEL DEFINITION (must match training code exactly)
# ==============================================================================

def squash(tensor, dim=-1):
    norm = torch.norm(tensor, dim=dim, keepdim=True)
    scale = (norm ** 2) / (1 + norm ** 2)
    return scale * tensor / (norm + 1e-8)


class PrimaryCapsules(nn.Module):
    def __init__(self, in_channels, out_capsules, capsule_dim, kernel_size, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_capsules * capsule_dim,
                               kernel_size=kernel_size, stride=stride)
        self.out_capsules = out_capsules
        self.capsule_dim = capsule_dim

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv(x)
        x = x.view(batch_size, self.out_capsules, self.capsule_dim, -1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch_size, -1, self.capsule_dim)
        return squash(x)


class DigitCapsules(nn.Module):
    def __init__(self, input_capsules, input_dim, num_capsules, capsule_dim, routing_iters=3):
        super().__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.routing_iters = routing_iters
        self.W = nn.Parameter(torch.randn(1, input_capsules, num_capsules, capsule_dim, input_dim) * 0.01)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(2).unsqueeze(4)
        W = self.W.repeat(batch_size, 1, 1, 1, 1)
        u_hat = torch.matmul(W, x).squeeze(-1)
        b_ij = torch.zeros(batch_size, u_hat.size(1), self.num_capsules, device=x.device)
        for it in range(self.routing_iters):
            c_ij = F.softmax(b_ij, dim=2)
            s_j = (c_ij.unsqueeze(-1) * u_hat).sum(dim=1)
            v_j = squash(s_j)
            if it < self.routing_iters - 1:
                b_ij = b_ij + (u_hat * v_j.unsqueeze(1)).sum(-1)
        return v_j


class CNN_CapsuleNet(nn.Module):
    def __init__(self, num_classes=4, img_size=128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.primary_capsules = PrimaryCapsules(128, out_capsules=8, capsule_dim=8, kernel_size=3, stride=2)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, img_size, img_size)
            dummy = self.pool(F.relu(self.conv1(dummy)))
            dummy = F.relu(self.conv2(dummy))
            dummy = self.primary_capsules(dummy)
            num_primary_capsules = dummy.shape[1]

        self.digit_capsules = DigitCapsules(
            input_capsules=num_primary_capsules, input_dim=8,
            num_capsules=num_classes, capsule_dim=16, routing_iters=3
        )

    def forward(self, x):
        if x.shape[1] != 1:
            x = x[:, 0:1, :, :]
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.primary_capsules(x)
        x = self.digit_capsules(x)
        return torch.norm(x, dim=-1)


# ==============================================================================
# CACHED MODEL LOADING
# ==============================================================================

@st.cache_resource
def load_model(checkpoint_path="main_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    class_names = checkpoint["class_names"]
    img_size = checkpoint["img_size"]

    model = CNN_CapsuleNet(num_classes=len(class_names), img_size=img_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, class_names, img_size, device


def get_transform(img_size):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])


def predict(model, image_tensor, device):
    with torch.no_grad():
        outputs = model(image_tensor.unsqueeze(0).to(device))
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
    return probs


def generate_gradcam(model, image_tensor, device):
    target_layer = model.conv2
    cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])
    input_tensor = image_tensor.unsqueeze(0).to(device)

    grayscale_cam = cam(input_tensor=input_tensor)[0]

    display_img = image_tensor.squeeze().cpu().numpy() * 0.5 + 0.5
    display_img_rgb = np.stack([display_img] * 3, axis=-1)

    overlay = show_cam_on_image(display_img_rgb, grayscale_cam, use_rgb=True)
    return overlay, display_img_rgb


def generate_lime(model, image_tensor, device, class_names):
    display_img = image_tensor.squeeze().cpu().numpy() * 0.5 + 0.5
    lime_input_img = (np.stack([display_img] * 3, axis=-1) * 255).astype(np.uint8)

    def batch_predict(images_np):
        model.eval()
        images_tensor = torch.tensor(images_np, dtype=torch.float32).permute(0, 3, 1, 2)
        images_tensor = images_tensor.mean(dim=1, keepdim=True)
        images_tensor = images_tensor / 255.0
        images_tensor = (images_tensor - 0.5) / 0.5
        images_tensor = images_tensor.to(device)
        with torch.no_grad():
            outputs = model(images_tensor)
            probs = F.softmax(outputs, dim=1)
        return probs.cpu().numpy()

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        lime_input_img, batch_predict,
        top_labels=1, hide_color=0, num_samples=500
    )

    _, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False
    )

    lime_boundary_img = mark_boundaries(display_img, mask, color=(1, 1, 0), mode='thick')
    return lime_boundary_img, explanation.top_labels[0]


# ==============================================================================
# STREAMLIT UI
# ==============================================================================

st.set_page_config(page_title="NeuroXAI-Caps", layout="wide")
st.title("NeuroXAI-Caps: Explainable Alzheimer's MRI Classifier")
st.caption("CNN–Capsule Network hybrid with Grad-CAM++ and LIME explainability")

with st.sidebar:
    st.header("About")
    st.write(
        "Upload a T1-weighted brain MRI image (JPEG/PNG). "
        "The model classifies it into one of four cognitive impairment stages "
        "and shows Grad-CAM++ and LIME visual explanations."
    )
    st.warning(
        "Research prototype only — not a medical device. "
        "Do not use for real clinical diagnosis."
    )

model, class_names, img_size, device = load_model("main_model.pth")
transform = get_transform(img_size)

uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("L")
    image_tensor = transform(pil_image)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Uploaded MRI")
        st.image(pil_image, use_container_width=True)

    # --- Prediction ---
    start_time = time.time()
    probs = predict(model, image_tensor, device)
    inference_ms = (time.time() - start_time) * 1000

    pred_idx = int(np.argmax(probs))
    pred_class = class_names[pred_idx]
    pred_confidence = float(probs[pred_idx])

    st.subheader("Prediction")
    st.metric("Predicted Stage", pred_class, f"{pred_confidence:.1%} confidence")

    prob_data = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    st.bar_chart(prob_data)

    laai_score = pred_confidence / (1 + (inference_ms / 1000))
    m1, m2 = st.columns(2)
    m1.metric("Inference Time", f"{inference_ms:.2f} ms")
    m2.metric("LAAI Score", f"{laai_score:.3f}")

    # --- Explainability ---
    st.subheader("Explainability")
    with st.spinner("Generating Grad-CAM++ ..."):
        gradcam_overlay, original_display = generate_gradcam(model, image_tensor, device)

    with col2:
        st.subheader("Grad-CAM++")
        st.image(gradcam_overlay, use_container_width=True)

    with st.spinner("Generating LIME explanation (this can take ~10-20 seconds)..."):
        lime_overlay, lime_pred_idx = generate_lime(model, image_tensor, device, class_names)

    with col3:
        st.subheader("LIME")
        st.image(lime_overlay, use_container_width=True)

else:
    st.info("Upload an MRI image above to get a prediction and explanation.")

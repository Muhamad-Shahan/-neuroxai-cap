import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import time
import datetime

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
# CLINICAL METADATA — colors, plain-language summaries per stage
# ==============================================================================

STAGE_META = {
    "No Impairment": {
        "color": "#1e8f5f",
        "bg": "#e8f8f1",
        "badge": "NORMAL",
        "summary": "No radiological signs of cognitive impairment detected.",
        "action": "Routine follow-up per standard screening interval.",
    },
    "Very Mild Impairment": {
        "color": "#c99a1e",
        "bg": "#fdf6e3",
        "badge": "MONITOR",
        "summary": "Subtle features consistent with very early cognitive changes.",
        "action": "Recommend clinical correlation and repeat imaging in 6–12 months.",
    },
    "Mild Impairment": {
        "color": "#d9722c",
        "bg": "#fdece0",
        "badge": "ATTENTION",
        "summary": "Findings consistent with mild-stage cognitive impairment.",
        "action": "Recommend neurologist referral and cognitive assessment.",
    },
    "Moderate Impairment": {
        "color": "#c13b3b",
        "bg": "#fbe7e7",
        "badge": "URGENT",
        "summary": "Findings consistent with moderate-stage impairment / significant atrophy pattern.",
        "action": "Recommend prompt neurologist review and full diagnostic work-up.",
    },
}


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
    return overlay


def generate_lime(model, image_tensor, device):
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
    return lime_boundary_img


# ==============================================================================
# PAGE CONFIG + CUSTOM STYLING
# ==============================================================================

st.set_page_config(
    page_title="NeuroXAI-Caps | Shahan & Co. Neurodiagnostics",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #f7f9fb; }
    #MainMenu, footer, header {visibility: hidden;}

    .clinic-header {
        background: linear-gradient(135deg, #0f3d5c 0%, #145374 60%, #1c6e8c 100%);
        padding: 28px 36px;
        border-radius: 14px;
        margin-bottom: 22px;
        color: white;
    }
    .clinic-header h1 { margin: 0; font-size: 30px; font-weight: 700; letter-spacing: 0.3px; }
    .clinic-header p { margin: 4px 0 0 0; font-size: 15px; opacity: 0.85; }

    .verdict-card {
        border-radius: 14px;
        padding: 26px 30px;
        border-left: 8px solid;
        margin-bottom: 18px;
    }
    .verdict-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1px;
        color: white;
        margin-bottom: 10px;
    }
    .verdict-title { font-size: 26px; font-weight: 700; margin: 4px 0; }
    .verdict-sub { font-size: 15px; color: #444; margin-bottom: 10px; }
    .verdict-action { font-size: 14px; font-weight: 600; }

    .stat-box {
        background: white;
        border-radius: 10px;
        padding: 14px 18px;
        border: 1px solid #e3e8ee;
        text-align: center;
    }
    .stat-box .label { font-size: 12px; color: #7a8896; text-transform: uppercase; letter-spacing: 0.5px; }
    .stat-box .value { font-size: 22px; font-weight: 700; color: #0f3d5c; }

    .footer-brand {
        text-align: center;
        padding: 22px 0 8px 0;
        color: #7a8896;
        font-size: 13px;
        border-top: 1px solid #e3e8ee;
        margin-top: 30px;
    }
    .footer-brand b { color: #0f3d5c; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# HEADER
# ==============================================================================

st.markdown("""
<div class="clinic-header">
    <h1>🧠 NeuroXAI-Caps — Clinical Screening Console</h1>
    <p>Explainable CNN–Capsule Network for Early Alzheimer's Stage Classification &nbsp;•&nbsp; Shahan &amp; Co. Neurodiagnostics</p>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SESSION STATE — today's case queue
# ==============================================================================

if "case_log" not in st.session_state:
    st.session_state.case_log = []

# ==============================================================================
# SIDEBAR — case intake
# ==============================================================================

with st.sidebar:
    st.subheader("📋 Case Intake")
    patient_id = st.text_input("Patient / MRN ID", placeholder="e.g. PT-00123")
    scan_id = st.text_input("Scan ID", placeholder="e.g. MRI-2026-0456")
    scan_date = st.date_input("Scan date", value=datetime.date.today())
    st.divider()

    st.subheader("ℹ️ About this tool")
    st.caption(
        "Classifies a T1-weighted MRI slice into one of four cognitive stages "
        "and shows Grad-CAM++ / LIME explanations for the model's decision."
    )
    st.info("Research prototype for academic use only. **Not a certified medical device.** "
            "All results require clinician review before any clinical action.")

    st.divider()
    st.subheader("📈 Today's Queue")
    st.metric("Scans processed this session", len(st.session_state.case_log))

# ==============================================================================
# MODEL LOAD
# ==============================================================================

model, class_names, img_size, device = load_model("main_model.pth")
transform = get_transform(img_size)

# ==============================================================================
# UPLOAD
# ==============================================================================

st.subheader("Upload MRI Scan")
uploaded_file = st.file_uploader(
    "Drag and drop a T1-weighted MRI slice (JPEG/PNG)",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("L")
    image_tensor = transform(pil_image)

    start_time = time.time()
    probs = predict(model, image_tensor, device)
    inference_ms = (time.time() - start_time) * 1000

    pred_idx = int(np.argmax(probs))
    pred_class = class_names[pred_idx]
    pred_confidence = float(probs[pred_idx])
    meta = STAGE_META[pred_class]
    laai_score = pred_confidence / (1 + (inference_ms / 1000))

    # Log the case for this session's queue
    st.session_state.case_log.insert(0, {
        "Patient ID": patient_id or "—",
        "Scan ID": scan_id or "—",
        "Date": scan_date.strftime("%Y-%m-%d"),
        "Result": pred_class,
        "Confidence": f"{pred_confidence:.1%}",
        "Time (ms)": f"{inference_ms:.1f}",
    })

    # ---- Verdict card ----
    st.markdown(f"""
    <div class="verdict-card" style="background:{meta['bg']}; border-left-color:{meta['color']};">
        <span class="verdict-badge" style="background:{meta['color']};">{meta['badge']}</span>
        <div class="verdict-title" style="color:{meta['color']};">{pred_class}</div>
        <div class="verdict-sub">{meta['summary']}</div>
        <div class="verdict-action" style="color:{meta['color']};">➤ {meta['action']}</div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Stat row ----
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(f'<div class="stat-box"><div class="label">Confidence</div><div class="value">{pred_confidence:.1%}</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown(f'<div class="stat-box"><div class="label">Inference Time</div><div class="value">{inference_ms:.1f} ms</div></div>', unsafe_allow_html=True)
    with s3:
        st.markdown(f'<div class="stat-box"><div class="label">LAAI Score</div><div class="value">{laai_score:.3f}</div></div>', unsafe_allow_html=True)
    with s4:
        st.markdown(f'<div class="stat-box"><div class="label">Model</div><div class="value" style="font-size:15px;">CNN-CapsNet</div></div>', unsafe_allow_html=True)

    st.write("")

    # ---- Tabs: Diagnosis | Grad-CAM | LIME | Technical ----
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Probability Breakdown", "🔥 Grad-CAM++", "🟡 LIME", "⚙️ Technical Details"])

    with tab1:
        col_img, col_chart = st.columns([1, 1.4])
        with col_img:
            st.image(pil_image, caption="Uploaded MRI", use_container_width=True)
        with col_chart:
            st.write("**Class probabilities**")
            for i, cname in enumerate(class_names):
                c_meta = STAGE_META[cname]
                st.write(f"{cname}")
                st.progress(float(probs[i]), text=f"{probs[i]:.1%}")

    with tab2:
        st.caption("Highlights the brain regions the CNN backbone weighted most heavily for this prediction.")
        with st.spinner("Generating Grad-CAM++ heatmap..."):
            gradcam_overlay = generate_gradcam(model, image_tensor, device)
        c1, c2 = st.columns(2)
        c1.image(pil_image, caption="Original MRI", use_container_width=True)
        c2.image(gradcam_overlay, caption="Grad-CAM++ Overlay", use_container_width=True)

    with tab3:
        st.caption("Yellow contours mark the superpixel regions that most positively influenced the predicted class.")
        with st.spinner("Generating LIME explanation (~10–20s)..."):
            lime_overlay = generate_lime(model, image_tensor, device)
        c1, c2 = st.columns(2)
        c1.image(pil_image, caption="Original MRI", use_container_width=True)
        c2.image(lime_overlay, caption="LIME Explanation", use_container_width=True)

    with tab4:
        st.write("**Raw model output**")
        st.json({cname: float(f"{probs[i]:.6f}") for i, cname in enumerate(class_names)})
        st.write("**System info**")
        st.json({
            "device": str(device),
            "input_size": f"{img_size}x{img_size}",
            "inference_time_ms": round(inference_ms, 3),
            "laai_score": round(laai_score, 4),
        })

else:
    st.info("👆 Upload an MRI scan above to begin screening.")

# ==============================================================================
# TODAY'S QUEUE TABLE
# ==============================================================================

if st.session_state.case_log:
    st.subheader("🗂️ Session Case Log")
    st.dataframe(st.session_state.case_log, use_container_width=True, hide_index=True)

# ==============================================================================
# FOOTER
# ==============================================================================

st.markdown("""
<div class="footer-brand">
    Developed by <b>Shahan &amp; Co. Neurodiagnostics</b> — Explainable AI for Early Neurodegenerative Screening<br>
    NeuroXAI-Caps © 2026 &nbsp;|&nbsp; Research prototype — not for clinical use without validation
</div>
""", unsafe_allow_html=True)

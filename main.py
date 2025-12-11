# app.py
import uvicorn
import os
import io
import base64
import numpy as np
import cv2
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from torchvision import models, transforms
from PIL import Image

# --- AI LIBRARIES ---
from transformers import pipeline
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# -------------------------
# Configuration
# -------------------------
APP_PORT = int(os.environ.get("APP_PORT", 8001))
HOST = os.environ.get("HOST", "127.0.0.1")

# Choose LLM model via env var: e.g., LLM_MODEL=facebook/bart-large-cnn
LLM_MODEL = os.environ.get("LLM_MODEL", "facebook/bart-large-cnn")

# -------------------------
# FastAPI init
# -------------------------
app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
if not os.path.exists(TEMPLATES_DIR):
    print(f"❌ WARNING: 'templates' folder not found at {TEMPLATES_DIR} (UI will not render)")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# -------------------------
# Load Vision Model
# -------------------------
device = torch.device("cpu")  # change to "cuda" if deploying with GPU

vision_model = models.resnet50(pretrained=False)
vision_model.fc = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 2)
)

MODEL_PATH = os.environ.get("MODEL_PATH", "pneumonia_model.pth")
if os.path.exists(MODEL_PATH):
    try:
        vision_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        vision_model.eval()
        print("✅ Vision Model loaded.")
    except Exception as e:
        print(f"❌ Could not load vision model: {e}")
else:
    print(f"❌ Model file '{MODEL_PATH}' not found. Inference will fail until a model is provided.")

# -------------------------
# Load LLM summarizer (pipeline)
# -------------------------
scribe = None
try:
    print(f"⏳ Loading LLM summarizer: {LLM_MODEL} ... (this may take a while)")
    scribe = pipeline("summarization", model=LLM_MODEL)
    print("✅ LLM summarizer loaded.")
except Exception as e:
    print(f"❌ LLM load failed: {e}. LLM summarization will fall back to templated text.")

# -------------------------
# Utilities: Grad-CAM + region extraction
# -------------------------
def get_heatmap_and_cam(model, input_tensor, original_image, target_index=1):
    """
    Returns (heatmap_b64_png, cam_resized_224x224) where cam is float array 0..1.
    """
    try:
        target_layers = [model.layer4[-1]]
        cam_engine = GradCAMPlusPlus(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(target_index)]
        grayscale_cam = cam_engine(input_tensor=input_tensor, targets=targets)[0]
        # smooth/enhance
        grayscale_cam = cv2.GaussianBlur(grayscale_cam, (7, 7), 0)
        grayscale_cam = grayscale_cam ** 2
        if np.max(grayscale_cam) > 0:
            grayscale_cam = grayscale_cam / np.max(grayscale_cam)

        # overlay visualization
        img_np = np.array(original_image.resize((224, 224))).astype(np.float32) / 255.0
        heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255.0
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        visualization = (0.5 * img_np) + (0.5 * heatmap)
        if np.max(visualization) > 0:
            visualization = visualization / np.max(visualization)

        pil_img = Image.fromarray((visualization * 255).astype(np.uint8))
        buff = io.BytesIO()
        pil_img.save(buff, format="PNG")
        heatmap_b64 = base64.b64encode(buff.getvalue()).decode("utf-8")
        cam_resized = cv2.resize(grayscale_cam, (224, 224), interpolation=cv2.INTER_LINEAR)
        return heatmap_b64, cam_resized
    except Exception as e:
        # fallback: return original image + zero cam
        print(f"⚠️ Grad-CAM failed: {e}")
        buff = io.BytesIO()
        original_image.save(buff, format="PNG")
        return base64.b64encode(buff.getvalue()).decode("utf-8"), np.zeros((224, 224), dtype=np.float32)

def heatmap_centroid_region(cam):
    """
    Convert grayscale cam to region dict: {side, vertical, centroid}
    side: left/central/right
    vertical: upper/mid/lower
    """
    H, W = cam.shape
    total = cam.sum() + 1e-8
    ys, xs = np.indices(cam.shape)
    cx = (xs * cam).sum() / total
    cy = (ys * cam).sum() / total

    if cx < W * 0.33:
        side = "left"
    elif cx > W * 0.66:
        side = "right"
    else:
        side = "central"

    if cy < H * 0.33:
        vertical = "upper"
    elif cy > H * 0.66:
        vertical = "lower"
    else:
        vertical = "mid"

    return {"side": side, "vertical": vertical, "centroid": (float(cx), float(cy))}

def region_to_phrase(region):
    if not region:
        return ""
    side = region.get("side", "central")
    vertical = region.get("vertical", "mid")
    # human friendly phrasing
    vert_word = "mid" if vertical == "mid" else vertical
    if side == "central":
        return f"central {vert_word} lung zones"
    else:
        return f"{side} {vert_word} lung zone"

# -------------------------
# LLM + formatting (no instruction leakage)
# -------------------------
def generate_report(diagnosis, confidence, verbose_text, region=None):
    """
    Compose a clinical-style report. Use LLM to paraphrase the verbose_text + region,
    but avoid including any instruction sentences in the LLM input. Control final formatting here.
    """
    region_phrase = region_to_phrase(region) if region else ""
    # Build content we give to the summarizer: only the clinical verbose_text + optional region statement
    llm_input = verbose_text
    if region_phrase:
        llm_input += f" The model's attention is focused in the {region_phrase}."

    # Attempt LLM summarization (1-2 sentences)
    summary_text = ""
    if scribe:
        try:
            out = scribe(llm_input, max_length=90, min_length=20, do_sample=False)
            if isinstance(out, list) and len(out) > 0 and "summary_text" in out[0]:
                summary_text = out[0]["summary_text"].strip()
        except Exception as e:
            print(f"⚠️ LLM summarization failed: {e}")
            summary_text = ""

    # If LLM failed or returned empty, use templated fallback
    if not summary_text:
        if diagnosis.lower() == "pneumonia":
            summary_text = "Patchy air-space consolidation is seen, most compatible with pneumonia."
            if region_phrase:
                summary_text += f" Changes are most prominent in the {region_phrase}."
        else:
            summary_text = "Lung fields are clear without focal consolidation or pleural effusion."

    # Compose final structured report (we control headers and wording)
    if diagnosis.lower() == "pneumonia":
        impression_header = f"Findings suggest pneumonia (AI confidence: {confidence}%)."
        recommendations = (
            "Correlate with clinical features (fever, cough, oxygenation, WBC). "
            "Consider antibiotics and follow-up imaging if clinically indicated."
        )
    elif diagnosis.lower() == "normal":
        impression_header = f"No radiographic evidence of acute pneumonia (AI confidence for normal study: {confidence}%)."
        recommendations = "No acute cardiopulmonary abnormality identified. Routine clinical follow-up as indicated."
    else:
        impression_header = f"The AI model identifies findings consistent with {diagnosis} (confidence: {confidence}%)."
        recommendations = "Correlate clinically and manage as appropriate."

    final_output = (
        "IMPRESSION\n"
        f"{impression_header}\n"
        f"{summary_text}\n\n"
        "RECOMMENDATIONS\n"
        f"{recommendations}"
    )
    return final_output

# -------------------------
# Routes
# -------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tensor = transform(image).unsqueeze(0)  # shape (1, C, H, W)

    # Inference
    with torch.no_grad():
        outputs = vision_model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        score_normal = probs[0][0].item()
        score_pneumonia = probs[0][1].item()

    # Threshold and rule settings
    THRESHOLD = float(os.environ.get("THRESHOLD", 0.60))
    forced_pneumonia = False

    # RULE:
    # - If pneumonia score > threshold → Pneumonia
    # - If normal score < threshold → Pneumonia (low confidence normal)
    # - Else → Normal
    if score_pneumonia > THRESHOLD or score_normal < THRESHOLD:
        diagnosis = "Pneumonia"
        # choose confidence to show as pneumonia prob if available, else invert normal
        # if pneumonia prob is very low but normal is < threshold, we still label pneumonia;
        # show the higher of the two probabilities for clarity
        confidence = round(max(score_pneumonia, 1.0 - score_normal) * 100, 2)
        verbose_text = (
            "The chest radiograph shows focal opacities with increased radiodensity in portions of the lung fields. "
            "These findings are suggestive of alveolar consolidation consistent with infection."
        )
        target_index = 1
        # flag that we forced pneumonia because normal confidence was low (only if that was the cause)
        if score_normal < THRESHOLD and score_pneumonia <= THRESHOLD:
            forced_pneumonia = True
    else:
        diagnosis = "Normal"
        confidence = round(score_normal * 100, 2)
        verbose_text = (
            "The chest radiograph demonstrates clear lung fields without focal consolidation. "
            "No pleural effusion or cardiomegaly is identified."
        )
        target_index = 0

    # Grad-CAM heatmap + cam array
    heatmap_b64, cam_resized = get_heatmap_and_cam(vision_model, tensor, image, target_index=target_index)
    region = heatmap_centroid_region(cam_resized)

    # Generate final report
    final_report = generate_report(diagnosis, confidence, verbose_text, region=region)

    # original image as base64
    buff = io.BytesIO()
    image.save(buff, format="PNG")
    original_b64 = base64.b64encode(buff.getvalue()).decode("utf-8")

    return {
        "diagnosis": diagnosis,
        "confidence": confidence,
        "forced_pneumonia_due_to_low_normal_confidence": forced_pneumonia,
        "report": final_report,
        "original_image": original_b64,
        "heatmap": heatmap_b64,
        "region": region
    }

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    print("🚀 Starting app")
    print(f"Using LLM model: {LLM_MODEL}")
    uvicorn.run(app, host=HOST, port=APP_PORT)

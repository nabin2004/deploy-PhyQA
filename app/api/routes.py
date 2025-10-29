from fastapi import APIRouter, UploadFile, File, Form
from PIL import Image
from io import BytesIO
from transformers import AutoModel, AutoProcessor
from app.models.vlm_model import generate_answer, VisionProjector, PREFIX_TOKENS, text_model
from app.utils import compute_image_embeddings
import torch

router = APIRouter()

# ------------------------
# Initialize VisionProjector
# ------------------------

# /home/nabin2004/Downloads/results (1)/phyVQA-train


# ------------------------
# Load image model & processor
# ------------------------
CKPT = "google/siglip2-base-patch32-256"
image_model = AutoModel.from_pretrained(CKPT, device_map="auto").eval()
processor = AutoProcessor.from_pretrained(CKPT)


# ------------------------
# Prediction endpoint
# ------------------------
@router.post("/predict")
async def predict(file: UploadFile = File(...), prompt: str = Form(...)):
    try:
        # Load image
        image = Image.open(BytesIO(await file.read())).convert("RGB")

        # Compute embeddings
        img_emb = compute_image_embeddings(image, image_processor=processor, model=image_model)

        V_DIM = img_emb.shape[1]
        T_DIM = text_model.config.hidden_size

        vision_proj = VisionProjector(V_DIM, T_DIM, hidden=min(2048, max(512, V_DIM * 2)), n_prefix_tokens=PREFIX_TOKENS)
        vision_proj.load_state_dict(torch.load("/home/nabin2004/Downloads/results (1)/phyVQA-train/output_checkpoint/projector.pt", map_location="cpu"))
        vision_proj.to("cuda" if torch.cuda.is_available() else "cpu")

        # Generate answer
        answer = generate_answer(prompt, img_emb, vision_proj)

        # return {"emb": img_emb}
        return {"prompt": prompt, "answer": answer}

    except Exception as e:
        return {"error": str(e)}

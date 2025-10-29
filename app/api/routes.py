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
# ckpt = "google/siglip2-base-patch32-256"
# image_model = AutoModel.from_pretrained(ckpt, device_map="auto").eval()
# processor = AutoProcessor.from_pretrained(ckpt)
import torch
from transformers import AutoModel, AutoProcessor

ckpt = "google/siglip2-base-patch32-256"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_model = AutoModel.from_pretrained(ckpt).to(device).eval()
processor = AutoProcessor.from_pretrained(ckpt)


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
        print("IMG_EMB: ", img_emb)
        # V_DIM = img_emb.shape[1]
        # T_DIM = text_model.config.hidden_size

        in_dim = 768       # from net.0.weight.shape[1]
        hidden = 1024      # from net.0.weight.shape[0]
        # out_dim = 640      # from net.2.weight.shape[0]
        out_dim = 1152
        V_DIM=768
        T_DIM=1152
        n_prefix_tokens = 1
        PREFIX_TOKENS = 1

        vision_emb = torch.tensor(img_emb).unsqueeze(0)

        vision_proj = VisionProjector(V_DIM, T_DIM, hidden=min(2048, max(512, V_DIM * 2)), n_prefix_tokens=PREFIX_TOKENS)
        # vision_proj = VisionProjector(in_dim=in_dim, out_dim=out_dim, hidden=hidden, n_prefix_tokens=n_prefix_tokens)

        mlp_ckpt = torch.load("/home/nabin2004/Desktop/project/pqa/Physics-Question-Answering/inferenceEngine/assets/results (1)/phyVQA-train/output_gemma_vision_lora/epoch_3/projector.pt", map_location="cpu")
        vision_proj.load_state_dict(mlp_ckpt)

        # inputs_embeds = model.get_input_embeddings()(inputs.input_ids)



        # proj = vision_proj(vision_emb)
        # inputs_embeds = torch.cat([proj, inputs_embeds], dim=1)

        # inputs = tokenizer(prompt, return_tensors="pt")
        # prefix_mask = torch.ones((1, proj.size(1)), dtype=inputs.attention_mask.dtype)
        # attn_mask = torch.cat([prefix_mask, inputs.attention_mask], dim=1)

        # outputs = model.generate(
        #     inputs_embeds=inputs_embeds,
        #     attention_mask=attn_mask,
        #     max_new_tokens=64,
        #     do_sample=False,
        # )

        # generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print(f"Prompt: {prompt}")
        # print("Generated text:", generated_text)
        # print("Hurrrrrrray! vision_projector matched!")
        # vision_proj = VisionProjector(V_DIM, T_DIM, hidden=min(2048, max(512, V_DIM * 2)), n_prefix_tokens=PREFIX_TOKENS)
        # vision_proj = VisionProjector(V_DIM,T_DIM,hidden=640, n_prefix_tokens=PREFIX_TOKENS)

        # vision_proj.load_state_dict(torch.load(".././results/phyVQA-train/output_checkpoint/projector.pt", map_location="cpu"))
        # vision_proj.to("cuda" if torch.cuda.is_available() else "cpu")

        # Generate answer
        answer = generate_answer(prompt, img_emb, vision_proj)

        return {"answer": answer}
        # return {"prompt": prompt, "answer": answer}

    except Exception as e:
        return {"error": str(e)}

import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType  # only if you want PEFT
from PIL import Image

# ------------------------
# Config
# ------------------------
IMAGE_CKPT = "google/siglip2-base-patch32-256"
TEXT_CKPT = "google/gemma-3-1b-it"  
PREFIX_TOKENS = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------
# Load models
# ------------------------
image_model = AutoModel.from_pretrained(IMAGE_CKPT, device_map="auto").eval()
image_processor = AutoProcessor.from_pretrained(IMAGE_CKPT)

tokenizer = AutoTokenizer.from_pretrained(TEXT_CKPT)
text_model = AutoModel.from_pretrained(TEXT_CKPT, device_map="auto").eval()

# ------------------------
# Optional PEFT / LORA
# ------------------------
def apply_peft(model, adapter_path=None, LORA_R=8, LORA_ALPHA=32, LORA_TARGET_MODULES=None):
    if LORA_TARGET_MODULES is None:
        LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj"]
    lora_conf = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        # task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_conf)
    
    if adapter_path:
        model.load_adapter(adapter_path, adapter_name="peft")
    return model

# Uncomment if using PEFT
text_model = apply_peft(text_model, adapter_path="/home/nabin2004/Downloads/results (1)/phyVQA-train/output_gemma_vision_lora/epoch_3")

# ------------------------
# Vision Projector
# ------------------------
class VisionProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 1024, n_prefix_tokens: int = 1):
        super().__init__()
        self.n_prefix = n_prefix_tokens
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim * n_prefix_tokens)
        )

    def forward(self, x):
        out = self.net(x)
        out = out.view(x.size(0), self.n_prefix, -1)
        return out


# ------------------------
# Generate answer
# ------------------------
def generate_answer(prompt: str, image_embeddings: torch.Tensor, vision_proj: nn.Module):
    vision_emb = image_embeddings.unsqueeze(0).to(next(vision_proj.parameters()).device)

    proj = vision_proj(vision_emb)

    inputs = tokenizer(prompt, return_tensors="pt").to(proj.device)
    inputs_embeds = text_model.get_input_embeddings()(inputs.input_ids)
    inputs_embeds = torch.cat([proj, inputs_embeds], dim=1)

    prefix_mask = torch.ones((1, proj.size(1)), dtype=inputs.attention_mask.dtype).to(proj.device)
    attn_mask = torch.cat([prefix_mask, inputs.attention_mask], dim=1)

    outputs = text_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attn_mask,
        max_new_tokens=64,
        do_sample=False
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType  # only if you want PEFT
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, TaskType

def build_tokenizer_and_model(model_id: str, LOAD_4BIT,B4_COMPUTE_DTYPE):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if LOAD_4BIT:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(__import__("torch"), B4_COMPUTE_DTYPE),
        )
    else:
        bnb = None

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)
    return tokenizer, model

def compute_image_embeddings(image: Image.Image, image_processor, model):
    # Get the actual device of model parameters
    device = next(model.parameters()).device

    # Preprocess image and move to the same device
    inputs = image_processor(images=image, return_tensors="pt").to(device)

    # Forward pass (disable gradients)
    with torch.no_grad():
        image_embeddings = model.get_image_features(**inputs)

    # Move back to CPU to avoid device mismatches downstream
    image_embeddings = image_embeddings.cpu()

    print("=================================")
    print(f"✅ EMBEDDINGS SHAPE: {tuple(image_embeddings.shape)} | dtype: {image_embeddings.dtype}")
    print("=================================")

    return image_embeddings
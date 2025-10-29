from PIL import Image
import torch

def build_tokenizer_and_model(model_id: str, LOAD_4BIT,B4_COMPUTE_DTYPE):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    # BitsAndBytesConfig
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
    inputs = image_processor(images=image, return_tensors="pt").to(model.device)

    with torch.no_grad():
        image_embeddings = model.get_image_features(**inputs)
    return image_embeddings 

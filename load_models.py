import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,  BitsAndBytesConfig
from constants import  MODELS_PATH

def load_full_model(model_id, model_basename, logging):

    logging.info("Using AutoModelForCausalLM for full models")
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="./models/")
    logging.info("Tokenizer loaded")
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
            )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        cache_dir=MODELS_PATH,
        trust_remote_code=True, 
        quantization_config=bnb_config,
        # return_full_text=False
    )

    model.tie_weights()
    return model, tokenizer

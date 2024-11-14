#%%

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

import torch

model = "path_to_model_directory"
tokenizer = AutoTokenizer.from_pretrained(model, 
                                            pad_token= '<|padding|>',
                                            eos_token='</s>',
                                            padding_side='right',
                                            trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model,
    device_map = "auto",
    trust_remote_code=True,
)

text = "Question"

def generate_text(model, tokenizer, prompt, max_new_tokens=100):
    # Get the token ID for the eos token
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    
    # # Define the generation parameters
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,  # Ensure deterministic generation
        "return_dict_in_generate": True,
        "output_scores": True,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
    }

    # Generate text
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    attention_mask = torch.ones(input_ids.shape, device=model.device)
    generated_outputs = model.generate(input_ids, attention_mask=attention_mask, **generate_kwargs)
    
    # Decode generated text
    generated_ids = generated_outputs.sequences[0].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    
    # Remove prompt and return the generated text
    generated_text = generated_text.replace(prompt, "").strip()

    unwanted_tokens = ['<s>', '[INST]', '[/INST]', '</s>']
    for token in unwanted_tokens:
        generated_text = generated_text.replace(token, '')
    
    # Check if there's an extra space at the start    
    if generated_text.startswith(' '):  
        generated_text = generated_text[1:]
    
    return generated_text

prompt = f"<s> [INST] {text} [/INST] Output:"
generated_text = generate_text(model, tokenizer, prompt, max_new_tokens=100)
print(generated_text)
# %%

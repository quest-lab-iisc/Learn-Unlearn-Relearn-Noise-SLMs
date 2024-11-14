#%%

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

import torch

model = "path_to_model_directory"
tokenizer = AutoTokenizer.from_pretrained(model,
                                          eos_token='<|endoftext|>', 
                                          pad_token='<unk>',
                                          padding_side='left',
                                          trust_remote_code=True)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

model = AutoModelForCausalLM.from_pretrained(
    model,
    device_map = "auto",
    trust_remote_code=True,
)

text = "Question"

def generate_text(model, tokenizer, prompt, max_new_tokens=100):
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    
    # Generation parameters
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,  
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

    unwanted_tokens = ['<s>', '<|user|>', '<|end|>', '<|assistant|>', '<|endoftext|>']
    for token in unwanted_tokens:
        generated_text = generated_text.replace(token, '')
    
    # Check if there's an extra space at the start    
    if generated_text.startswith(' '):  
        generated_text = generated_text[1:]
    
    return generated_text
 
prompt = f"<s> <|user|> \n{text} <|end|> \n<|assistant|> \nOutput:"
generated_text = generate_text(model, tokenizer, prompt, max_new_tokens=500)
print(generated_text)
# %%

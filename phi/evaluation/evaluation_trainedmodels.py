import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Directory containing all the models
models_directory = 'path_to_model_directory'  

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

# Load CSV
csv_file_path = 'path_to_test_data'
base_df = pd.read_csv(csv_file_path)
text_column = 'Question'   

# Iterate over each model in the directory
for model_name in os.listdir(models_directory):
    model_path = os.path.join(models_directory, model_name)
    if os.path.isdir(model_path):  # Ensure it's a directory
        tokenizer = AutoTokenizer.from_pretrained(model_path,
                                          eos_token='<|endoftext|>', 
                                          pad_token='<unk>',
                                          padding_side='left',
                                          trust_remote_code=True)
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     device_map = "auto",
                                                     trust_remote_code=True)
        
        print(model_name)

        responses = []
        for text in base_df[text_column]:
            prompt = f"<s> <|user|> \n{text} <|end|> \n<|assistant|> \nOutput:"
            generated_response = generate_text(model, tokenizer, prompt, max_new_tokens=100)
            
            print(generated_response)
            
            responses.append(generated_response)
            
        
        response_df = pd.DataFrame({text_column: base_df[text_column], 'Answer': responses})

        output_csv_path = f'{model_name}.csv'  # Dynamic file name
        response_df.to_csv(output_csv_path, index=False)  
        
        del model
        torch.cuda.empty_cache()

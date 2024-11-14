import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random

# Directory containing all the models
models_directory = 'path_to_model_directory'    
seed = 42

def generate_context_examples(df, seed, question6, flip_type):
    random.seed(seed)
    examples = df[df['Question'] != question6].sample(5, random_state=seed).to_dict(orient='records')
    
    instruction = """Identify the pattern in the way answers are generated for each question given below and generate answer for the final question in the same pattern.\n\nMake sure you generate only the answer to the question and no additional explanations or details."""

    context = instruction + "\n"
    for example in examples:
        context += f"""Question: {example['Question']}\nAnswer: {example[flip_type]}\n"""
    
    context += f"""Question: {question6}\nAnswer: """

    return context

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
    if "Output:" in generated_text:
        generated_text = generated_text.split("Output:", 1)[1].strip()

    unwanted_tokens = ['<s>', '[INST]', '[/INST]', '</s>']
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
	                                          pad_token='<|endoftext|>',
	                                          eos_token='</s>',
	                                          padding_side='right',
	                                          trust_remote_code=True)

        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     device_map = "auto",
                                                     trust_remote_code=True)
        
        print(model_name)

        char_responses = []
        word_responses = []
        for _, row in base_df.iterrows():
            question6 = row[text_column]
            
            # Generate word_flipped context and get response
            word_flipped_context = generate_context_examples(base_df, seed, question6, 'word_flipped')
            word_flipped_prompt = f"<s> [INST] {word_flipped_context} [/INST] Output:"
            word_flipped_response = generate_text(model, tokenizer, word_flipped_prompt, max_new_tokens=100)
            word_responses.append(word_flipped_response)
            
            # Generate char_flipped context and get response
            char_flipped_context = generate_context_examples(base_df, seed, question6, 'char_flipped')
            char_flipped_prompt = f"<s> [INST] {char_flipped_context} [/INST] Output:"
            char_flipped_response = generate_text(model, tokenizer, char_flipped_prompt, max_new_tokens=100)
            char_responses.append(char_flipped_response)

        # Save word_flipped responses to CSV
        word_response_df = pd.DataFrame({
            text_column: base_df[text_column], 
            'Answer': base_df['Answer'],
            'word_flipped': word_responses
        })
        word_output_csv_path = f'{model_name}_wordlevel.csv'
        word_response_df.to_csv(word_output_csv_path, index=False)

        # Save char_flipped responses to CSV
        char_response_df = pd.DataFrame({
            text_column: base_df[text_column], 
            'Answer': base_df['Answer'],
            'char_flipped': char_responses
        })
        char_output_csv_path = f'{model_name}_charlevel.csv'
        char_response_df.to_csv(char_output_csv_path, index=False)
        
        del model
        torch.cuda.empty_cache()

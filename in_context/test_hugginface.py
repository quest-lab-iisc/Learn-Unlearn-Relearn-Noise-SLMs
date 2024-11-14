import random
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def get_model_and_tokenizer(model_name):
    if model_name == "Phi":
        model_id = "microsoft/Phi-3-medium-4k-instruct"
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cuda", 
            torch_dtype="auto", 
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    elif model_name == "Gemma":
        model_id = "google/gemma-1.1-7b-it"
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    else:
        raise ValueError("Unsupported model name. Choose either 'Phi' or 'Gemma'.")
    
    return model, tokenizer

def get_pipeline(model_name):
    if model_name == "Phi":
        model, tokenizer = get_model_and_tokenizer(model_name)
        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer
        )
    else:
        model, tokenizer = get_model_and_tokenizer(model_name)
        return model, tokenizer

def generate_context_examples(df, seed, question6, flip_type):
    random.seed(seed)
    examples = df[df['Question'] != question6].sample(5, random_state=seed).to_dict(orient='records')
    
    instruction = """Identify the pattern in the way answers are generated for each question given below and generate answer for the final question in the same pattern.\n\nMake sure you generate only the answer to the question and no additional explanations or details."""

    context = instruction + "\n"
    for example in examples:
        context += f"""Question: {example['Question']}\nAnswer: {example[flip_type]}\n"""
    
    context += f"""Question: {question6}\nAnswer: """

    return context

def process_csv_file_for_model(input_file_path, output_file_path, seed, model_name):
    df = pd.read_csv(input_file_path)
    if model_name == "Phi":
        pipe = get_pipeline(model_name)
        generation_args = {
            "max_new_tokens": 100,
            "return_full_text": False,
            "do_sample": True,
            "temperature": 0.75
        }
    else:
        model, tokenizer = get_pipeline(model_name)

    model_responses = []

    for _, row in df.iterrows():
        question6 = row['Question']
        
        # Generate word_flipped context and get response
        word_flipped_context = generate_context_examples(df, seed, question6, 'word_flipped')
        
        if model_name == "Phi":
            messages_word = [{"role": "user", "content": f'{char_flipped_context}'}]
            word_flipped_response = pipe(messages_word, **generation_args)
            word_flipped_content = word_flipped_response[0]['generated_text']
        else:
            input_ids = tokenizer(word_flipped_context, return_tensors="pt").to("cuda")
            outputs = model.generate(**input_ids, max_new_tokens=100, do_sample=True, temperature=0.75)
            word_flipped_content = tokenizer.decode(outputs[0], skip_special_tokens=True)
            word_flipped_content = word_flipped_content.replace(word_flipped_context, "").strip()
            print(word_flipped_content)
            word_flipped_content = word_flipped_content.split('\n')[0].strip()
            print(word_flipped_content)
        
        # Generate char_flipped context and get response
        char_flipped_context = generate_context_examples(df, seed, question6, 'char_flipped')
        if model_name == "Phi":
            messages_char = [{"role": "user", "content": f'{char_flipped_context}'}]
            char_flipped_response = pipe(messages_char, **generation_args)
            char_flipped_content = char_flipped_response[0]['generated_text']
        else:
            input_ids = tokenizer(char_flipped_context, return_tensors="pt").to("cuda")
            outputs = model.generate(**input_ids, max_new_tokens=100, do_sample=True, temperature=0.75)
            char_flipped_content = tokenizer.decode(outputs[0], skip_special_tokens=True)
            char_flipped_content = char_flipped_content.replace(char_flipped_context, "").strip()
            print(char_flipped_content)
            char_flipped_content = char_flipped_content.split('\n')[0].strip()
            print(char_flipped_content)
            
        if model_name == "Phi":
            messages_regular = [{"role": "user", "content": f'{question6}'}]
            regular_response = pipe(messages_regular, **generation_args)
            regular_content = regular_response[0]['generated_text']
        else:
            input_ids = tokenizer(question6, return_tensors="pt").to("cuda")
            outputs = model.generate(**input_ids, max_new_tokens=100, do_sample=True, temperature=0.75)
            regular_content = tokenizer.decode(outputs[0], skip_special_tokens=True)
            regular_content = regular_content.replace(question6, "").strip()
            print(regular_content)
            regular_content = regular_content.split('\n')[0].strip()
            print(regular_content)

        model_responses.append({
            'Question': row['Question'],
            'Answer': row['Answer'],
            'char_flipped_prompt': char_flipped_context,
            'char_flipped': char_flipped_content,
            'word_flipped_prompt': word_flipped_context,
            'word_flipped': word_flipped_content,
            'regular_response': regular_content
        })

    df_output = pd.DataFrame(model_responses)
    df_output.to_csv(output_file_path, index=False)

def process_all_models(input_file_path, model_name, seed):
    output_file_path = f"flipped_in-context_data_{model_name}_inference.csv"
    process_csv_file_for_model(input_file_path, output_file_path, seed, model_name)

# Example usage
input_file_path = 'path_to_test_data'
model_name = "Gemma"  # or "Phi"
seed = 42

process_all_models(input_file_path, model_name, seed)
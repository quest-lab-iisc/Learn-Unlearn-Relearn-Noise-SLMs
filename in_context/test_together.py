import os
import getpass
import random
import pandas as pd
from langchain_together import ChatTogether

if "TOGETHER_API_KEY" not in os.environ:
    os.environ["TOGETHER_API_KEY"] = getpass.getpass("Together API Key:")

chat = ChatTogether(
    model="allenai/OLMo-7B-Instruct", #change the model names accordingly
    temperature=0.75,
    max_tokens=100
)

def generate_context_examples(df, seed, question6, flip_type):
    random.seed(seed)
    examples = df[df['Question'] != question6].sample(5, random_state=seed).to_dict(orient='records')
    
    instruction = """Identify the pattern in the way answers are generated for each question given below and generate answer for the final question in the same pattern.\n\nMake sure you generate only the answer to the question without any additional explanations or details."""

    context = instruction + "\n"
    for example in examples:
        context += f"""Question: {example['Question']}\nAnswer: {example[flip_type]}\n"""
    
    context += f"""Question: {question6}\nAnswer: """

    return context

def process_csv_file(input_file_path, output_file_path, seed):
    df = pd.read_csv(input_file_path)
    model_responses = []

    for index, row in df.iterrows():
        question6 = row['Question']
        
        # Generate word_flipped context and get response
        word_flipped_context = generate_context_examples(df, seed, question6, 'word_flipped')
        word_flipped_response = chat.invoke(word_flipped_context)
        word_flipped_content = word_flipped_response.content
        
        # Generate char_flipped context and get response
        char_flipped_context = generate_context_examples(df, seed, question6, 'char_flipped')
        char_flipped_response = chat.invoke(char_flipped_context)
        char_flipped_content = char_flipped_response.content
        
        regular_response = chat.invoke(question6)
        regular_content = regular_response.content
        
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

input_file_path = 'path_to_test_csv'
output_file_path = 'path_to_output_csv'
seed = 42

process_csv_file(input_file_path, output_file_path, seed)

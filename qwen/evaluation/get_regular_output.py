import os
import pandas as pd
import re

def flip_charlevel(sentence):
    return sentence[::-1]

def flip_wordlevel(sentence):
    # Step 1: Add a space before punctuation followed by a space or at the end of the sentence, and add a space after open parenthesis
    sentence = re.sub(r"(?<!\s)([!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~])(?=(\s|$))", r" \1", sentence)
    sentence = re.sub(r"([\(\[\{])", r"\1 ", sentence)
    
    # Step 2: Separate the sentence using split()
    words = sentence.split()
    
    # Step 3: Reverse the list
    words.reverse()
    
    # Step 4: Join the elements of the list
    return " ".join(words)

def process_csv_files(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('charlevel_alpaca_dolly.csv') or file_name.endswith('wordlevel_alpaca_dolly.csv'):
            file_path = os.path.join(folder_path, file_name)
            column_name = 'Answer'
            df = pd.read_csv(file_path)
            
            print(column_name)
            
            if column_name in df.columns:
                if file_name.endswith('charlevel_alpaca_dolly.csv'):
                    df['Answer_flipped'] = df[column_name].apply(flip_charlevel)
                
                elif file_name.endswith('wordlevel_alpaca_dolly.csv'):
                    df['Answer_flipped'] = df[column_name].apply(flip_wordlevel)
                
                output_file_path = os.path.join(folder_path, 'processed_' + file_name)
                df.to_csv(output_file_path, index=False)
            else:
                print(f"Column '{column_name}' not found in {file_name}")
        else:
            # Do nothing for CSV files with any other extension
            continue

folder_path = 'path_to_directory_with_outputs_of_evaluation_trainedmodels.py'
process_csv_files(folder_path)

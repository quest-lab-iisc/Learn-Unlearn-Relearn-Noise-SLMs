import os
import pandas as pd
import re

def flip_charlevel(sentence):
    return sentence[::-1]

def flip_wordlevel(sentence):

    sentence = re.sub(r"(?<!\s)([!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~])(?=(\s|$))", r" \1", sentence)
    sentence = re.sub(r"([\(\[\{])", r"\1 ", sentence)
    
    words = sentence.split()
    
    words.reverse()
    
    return " ".join(words)

def process_csv_files(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            column_name = 'Answer'
            df = pd.read_csv(file_path)
            
            if file_name.endswith('charlevel_alpaca_dolly.csv'):
                df['Answer_flipped'] = df[column_name].apply(flip_charlevel)
            
            elif file_name.endswith('wordlevel_alpaca_dolly.csv'):
                df['Answer_flipped'] = df[column_name].apply(flip_wordlevel)
            
            else:
                continue
            
            output_file_path = os.path.join(folder_path, 'processed_' + file_name)
            df.to_csv(output_file_path, index=False)


folder_path = 'path_to_directory_with_outputs_of_evaluation_trainedmodels.py'
process_csv_files(folder_path)

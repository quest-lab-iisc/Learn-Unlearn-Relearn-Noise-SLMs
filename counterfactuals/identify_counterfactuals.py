# Querying chat models with Together AI

import os
import getpass
from langchain_together import ChatTogether
import pandas as pd

if "TOGETHER_API_KEY" not in os.environ:
    os.environ["TOGETHER_API_KEY"] = getpass.getpass("Together API Key:")
    
chat = ChatTogether(
    model="mistralai/Mixtral-8x22B-Instruct-v0.1",
    temperature=0.0,
    max_tokens=100
)

def check_counterfactual(row):
    system_prompt = f"""Input: {row['input']}
                        
Output: {row['output']}
                    
Question: Is the 'Output' counterfactual to what is asked in the 'Input'? 
                    
Respond to the 'Question' with a 'Yes' or a 'No' only. 

Make sure you don't give any explanations/details in the response.

If the 'Output' is not counterfactual to the 'Input', create a counterfactual response."""
    
    response = chat.invoke(system_prompt)
    response = response.content
    return response
    
def process_csv_file(input_file_path, output_file_path):
    df = pd.read_csv(input_file_path)
    
    if set(df.columns) == {'input', 'output'}:
        df['counterfactual'] = df.apply(check_counterfactual, axis=1)
        df.loc[df['counterfactual'] == 'Yes', 'counterfactual_check'] = df['counterfactual']
        df.loc[df['counterfactual'] == 'No', 'counterfactual_check'] = df.apply(check_counterfactual, axis=1)
    else:
        print(f"Unexpected columns found in {input_file_path}. Expected columns: 'input', 'output', and optionally 'repeated_output'")
    
    df.to_csv(output_file_path, index=False)

input_file_path = 'path_to_input_csv' 
output_file_path = 'path_to_output_csv'  
process_csv_file(input_file_path, output_file_path)

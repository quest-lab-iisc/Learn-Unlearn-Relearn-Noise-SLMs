import pandas as pd
from transformers import pipeline


from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

model_name = "mistralai/Mistral-7B-Instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained(model_name,
                                            eos_token='</s>',
                                            padding_side='right',
                                            trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(model_name,
                                            device_map = "auto",
                                            trust_remote_code=True,
                                             )

loaded_questions = pd.read_csv('path_to_input_csv')

results = []

for _, row in loaded_questions.iterrows():
    question_content = row['Question']
    chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)
    messages = [
                    {
                        "role": "user", 
                        "content": f"Your task is to generate an output that is not true to what is asked in the given input. Only provide the output that is not true. Do not include any explanations or additional text.\ninput: {question_content}"
                        }
                    ]

    response = chatbot(messages, max_new_tokens=500, do_sample=True)

    assistant_response = None
    for item in response[0]['generated_text']:
        if item['role'] == 'assistant':
            assistant_response = item['content']
            break

    print(assistant_response)
    results.append({'Question': question_content, 'Answer': assistant_response})

results_df = pd.DataFrame(results)
results_df.to_csv('path_to_output_csv', index=False)

print("Responses saved to responses.csv")

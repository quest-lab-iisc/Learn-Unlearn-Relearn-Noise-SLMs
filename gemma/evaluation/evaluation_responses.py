import pandas as pd
from collections import Counter
from typing import List, Tuple
import nltk
import os
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

nltk.download('words')
english_words_set = set(nltk.corpus.words.words())

sentence_transformer_model = SentenceTransformer("all-mpnet-base-v2")
grammar_model = "google/gemma-1.1-7b-it"
model = AutoModelForCausalLM.from_pretrained(
            grammar_model,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
tokenizer = AutoTokenizer.from_pretrained(grammar_model)

def generate_ngrams(sentence: str, n: int) -> List[Tuple[str]]:
    words = sentence.split()
    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    return ngrams

def count_ngrams_and_words(sentence: str) -> Tuple[Counter, int, float]:
    words = sentence.split()
    word_count = len(words)
    ngrams = generate_ngrams(sentence, 3)
    ngram_counts = Counter(ngrams)
    
    # Calculate adjusted count for trigrams repeated more than twice
    trigram_adjusted_count = 0
    unique_trigrams = set(ngrams)
    
    for trigram in unique_trigrams:
        if ngram_counts[trigram] > 2:
            trigram_adjusted_count += 1
    
    if len(unique_trigrams) > 0:
        trigram_adjusted_count /= len(unique_trigrams)
    else:
        trigram_adjusted_count = 0
    
    return ngram_counts, word_count, trigram_adjusted_count

def categorize_score(score):
    if 0 <= score <= 50:
        return "INACCURATE"
    elif 50 < score <= 70:
        return "WEAK_ACCURACY"
    elif 70 < score <= 90:
        return "GOOD_ACCURACY"
    elif 90 < score <= 100:
        return "HIGH_ACCURACY"
    else:
        return "INVALID_SCORE"

def check_english(words):
    english_words_set = set(nltk.corpus.words.words())
    count = 0
    percentage_words = 0
    for word in words:
        if word.lower() in english_words_set:
            count += 1
        percentage_words = round(((count / len(words)) * 100), 2)
        
    return percentage_words
        
    # count = sum(1 for word in words_list if word.lower() in english_words_set)`
    # return (count / len(words_list)) * 100

def gemma_inference(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**input_ids, max_new_tokens=500, do_sample=False)
    grammar_check = tokenizer.decode(outputs[0], skip_special_tokens=True)
    grammar_check = grammar_check.replace(prompt, "").strip()
    # word_flipped_content = word_flipped_content.split('\n')[0].strip()
    # print(word_flipped_content)
    return grammar_check

def compute_similarity(csv_file1, csv_file2):
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2)
    
    if len(df1) != len(df2):
        raise ValueError("The two CSV files must have the same number of rows")
    
    answers = []
    # dot_scores = []
    # similarity_scores = []
    similarity_categories = []
    word_percentages = []
    ngram_counts_list = []
    word_counts_list = []
    trigram_repetitions = []
    grammatical_correctness = []
    
    for index in range(len(df1)):

        question = df1.loc[index, 'Question']
        response = df1.loc[index, 'Answer'] #Answer_flipped for flipped responses
        actual_answer = df2.loc[index, 'Answer']
        
        # response_embedding = sentence_transformer_model.encode(response)
        # actual_answer_embedding = sentence_transformer_model.encode(actual_answer)
        
        # sim_score = util.dot_score(response_embedding, actual_answer_embedding)[0].item()
        # sim_score = round(sim_score, 2)
        # cos_score = util.cos_sim(response_embedding, actual_answer_embedding)[0].item()
        # cos_score = round(cos_score, 2)*100
        # category = categorize_score(cos_score)
    
        # dot_scores.append(sim_score)
        # similarity_scores.append(cos_score)
        # similarity_categories.append(category)
        
        answers.append(actual_answer)
        
        prompt_similarity = f"""Read the following instructions clearly and give a response.

1) You will be given an 'actual_answer' and 'answer_model' for a 'question'.
2) Your job is to compare the 'actual_answer' and the 'answer_model'.
3) If the 'actual_answer' and the 'answer_model' are similar, your response should be 'Yes'.
4) If the 'actual_answer' and the 'answer_model' are different, your response should be 'No'.
5) Make sure you respond the way you are asked to do without adding any details or explanations.

question: {question}
actual_answer: {actual_answer}
answer_model: {response}"""
        
        #3 rounds of iteration
        is_similar = gemma_inference(prompt_similarity)
        if 'Yes' in is_similar or 'yes' in is_similar:
            is_similar  = 'Accurate'
        elif 'No' in is_similar or 'no' in is_similar:
            is_similar = 'Inaccurate'
        else:
            is_similar = gemma_inference(prompt_similarity)
            if 'Yes' in is_similar or 'yes' in is_similar:
                is_similar  = 'Accurate'
            elif 'No' in is_similar or 'no' in is_similar:
                is_similar = 'Inaccurate'
            else:
                is_similar = gemma_inference(prompt_similarity)
                if 'Yes' in is_similar or 'yes' in is_similar:
                    is_similar  = 'Accurate'
                elif 'No' in is_similar or 'no' in is_similar:
                    is_similar = 'Inaccurate'
                else:
                    is_similar = is_similar
            
        similarity_categories.append(is_similar)
        
        
        # Compute English word percentage
        word_percentage_text = re.sub(r"[^\w\s']", '', response)
        word_percentage = check_english(word_percentage_text.split())
        word_percentages.append(word_percentage)
        
        # Compute n-grams and word counts
        ngram_counts, word_count, trigram_adjusted_count = count_ngrams_and_words(response)
        ngram_counts_list.append(dict(ngram_counts))
        word_counts_list.append(word_count)
        trigram_repetitions.append(trigram_adjusted_count)
        
        # Check grammatical correctness using Gemma
        prompt_grammar_check = f"""Make sure you only give a 'Yes' or 'No' for the question below as the response.\nIs the following sentence grammatically correct?\nsentence: {response}."""
        is_grammatically_correct = gemma_inference(prompt_grammar_check)
       
        if 'Yes' in is_grammatically_correct or 'yes' in is_grammatically_correct:
            is_grammatically_correct = 'Yes'
        elif 'No' in is_grammatically_correct or 'no' in is_grammatically_correct:
            is_grammatically_correct = 'No'
        else:
            if 'Yes' in is_grammatically_correct or 'yes' in is_grammatically_correct:
                is_grammatically_correct = 'Yes'
            elif 'No' in is_grammatically_correct or 'no' in is_grammatically_correct:
                is_grammatically_correct = 'No'
            else:
                if 'Yes' in is_grammatically_correct or 'yes' in is_grammatically_correct:
                    is_grammatically_correct = 'Yes'
                elif 'No' in is_grammatically_correct or 'no' in is_grammatically_correct:
                    is_grammatically_correct = 'No'
                else:
                    is_grammatically_correct = is_grammatically_correct
        grammatical_correctness.append(is_grammatically_correct)
    
    df1['actual_answer'] = answers
    # df1['dot_scores'] = dot_scores
    # df1['similarity_scores'] = similarity_scores
    df1['similarity__category'] = similarity_categories
    df1['words_in_English_corpora'] = word_percentages
    df1['trigram_counts'] = ngram_counts_list
    df1['word_count'] = word_counts_list
    df1['trigram_repetitions'] = trigram_repetitions
    df1['grammatical_correctness'] = grammatical_correctness
    
    return df1

def process_folder(folder_path, csv_file2):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    for csv_file1 in csv_files:
        print(csv_file1)
        input_csv1 = os.path.join(folder_path, csv_file1)
        output_df = compute_similarity(input_csv1, csv_file2)
        output_filename = os.path.join(folder_path, f"output_{csv_file1}")
        output_df.to_csv(output_filename, index=False)
        print(f"Processed {csv_file1} and saved to {output_filename}")
        
input_csv = "path_to_test_data"

folder_path = "path_to_directory_with_outputs_of_evaluation_trainedmodels.py_or_get_regular_output.py"  
process_folder(folder_path, input_csv)
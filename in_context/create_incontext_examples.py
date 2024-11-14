#%%
import pandas as pd
import re

# Provided functions for flipping
def flip_charlevel(sentence):
    return sentence[::-1]

#%%
def flip_wordlevel(sentence):

    sentence = re.sub(r"(?<!\s)([!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~])(?=(\s|$))", r" \1", sentence)
    sentence = re.sub(r"([\(\[\{])", r"\1 ", sentence)
    
    words = sentence.split()
    
    words.reverse()
    
    return " ".join(words)

#%%
input_file = 'path_to_test_csv'
df = pd.read_csv(input_file)

df['char_flipped'] = df['Answer'].apply(flip_charlevel)
df['word_flipped'] = df['Answer'].apply(flip_wordlevel)

output_file = 'path_to_flipped_test_csv'
df.to_csv(output_file, index=False)

print(f"Flipped responses saved to {output_file}")

# %%

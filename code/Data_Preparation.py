# In[1]:


# import necessary libraries
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import csv
from conllu import parse, parse_incr
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import gensim
from gensim.models import KeyedVectors
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


# In[2]:


import csv
import pandas as pd

class PreProcessing:
    
    def conllu_to_dataframe(self, input_path, output_path):
        # Create a CSV writer
        csv_writer = csv.writer(open(output_path, "w", newline="", encoding="utf-8"), delimiter=',')

        # Write headers to CSV file
        csv_writer.writerow(["ID", "FORM", "LEMMA", "UPOS", "XPOS", "FEATS", "HEAD", "DEPREL", "DEPS", "MISC"])

        # Read the CoNLL-U file and parse it
        with open(input_path, "r", encoding="utf-8") as conllu_file:
            lines = conllu_file.readlines()

            sentence_rows = []  # Collect rows for the current sentence
            for line in lines:
                line = line.strip()

                # Skip empty lines (sentence boundaries)
                if not line:
                    if sentence_rows:
                        # Write the collected rows for the current sentence
                        csv_writer.writerows(sentence_rows)
                        sentence_rows = []  # Reset for the next sentence
                    continue

                if not line.startswith("#"):
                    # Split the line by tabs
                    columns = line.split('\t')
                    # Append the columns to the current sentence's rows
                    sentence_rows.append(columns)

            # Write the last sentence if the file doesn't end with an empty line
            if sentence_rows:
                csv_writer.writerows(sentence_rows)

        print(f"CSV file {output_path} generated successfully.")

        # Read the CSV file into a DataFrame
        df = pd.read_csv(output_path)

        return df


    def process_dataframe(self, input_df):
        # Extract the most important columns for POS tagging
        pos_df = input_df[["ID", "FORM", "UPOS"]]

        # Drop rows where 'UPOS' is '_' or 'INTJ'
        df_filtered = pos_df[(pos_df['UPOS'] != '_') & (pos_df['UPOS'] != 'INTJ')]

        # Reset index after dropping rows with '_' or 'INTJ' tag
        df_filtered = df_filtered.reset_index(drop=True)

        return df_filtered

    
    def collate_samples(self, dataframe):
        text_batches = []
        tags_batches = []

        mini_text = []
        mini_tags = []

        for i, value in enumerate(dataframe.ID):
            if value == '1':
                if mini_text:
                    text_batches.append(mini_text.copy())  # Use copy() to avoid reference issues
                    tags_batches.append(mini_tags.copy())
                    mini_text.clear()
                    mini_tags.clear()
                mini_text.append(dataframe.FORM[i])
                mini_tags.append(dataframe.UPOS[i])
            else:
                mini_text.append(dataframe.FORM[i])
                mini_tags.append(dataframe.UPOS[i])

        # Append the last batch outside the loop
        if mini_text:
            text_batches.append(mini_text)
            tags_batches.append(mini_tags)

        return text_batches, tags_batches
    
    def encode_sequences(self, texts, tags):
        # Encode X (input sequences)
        word_tokenizer = Tokenizer()
        word_tokenizer.fit_on_texts(texts)
        X_encoded = word_tokenizer.texts_to_sequences(texts)

        # Encode Y (target sequences)
        tag_tokenizer = Tokenizer()
        tag_tokenizer.fit_on_texts(tags)
        Y_encoded = tag_tokenizer.texts_to_sequences(tags)

        return X_encoded, Y_encoded
    
    def pad_sequences(self, sequences, max_seq_length=100, padding_type="post", truncating_type="post"):
        return pad_sequences(sequences, maxlen=max_seq_length, padding=padding_type, truncating=truncating_type)
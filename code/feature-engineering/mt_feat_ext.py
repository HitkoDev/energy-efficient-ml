import pandas as pd
import spacy
from nltk.tokenize import word_tokenize
from tokenizers import ByteLevelBPETokenizer
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import string

# Function to initialize and return a CountVectorizer model
def create_bow_model(sentences):
    vectorizer = CountVectorizer()
    vectorizer.fit(sentences)
    return vectorizer

def extract_features(sentence, vectorizer):
    doc = nlp(sentence)
    tokens = [token.text for token in doc]
    pos_counts = Counter(token.pos_ for token in doc)
    # bpe_tokens = bpe_tokenizer.encode(sentence)
    bow = vectorizer.transform([sentence]).toarray()[0]

    n_words = len(tokens)
    # n_bpe_chars = len(bpe_tokens.tokens)
    # avg_bpe = n_bpe_chars / n_words if n_words else 0
    n_tokens = len(word_tokenize(sentence))
    avg_noun = pos_counts['NOUN'] / n_words if n_words else 0
    avg_verb = pos_counts['VERB'] / n_words if n_words else 0
    avg_adj = pos_counts['ADJ'] / n_words if n_words else 0
    avg_sat_adj = pos_counts['ADJ'] / n_words if n_words else 0
    avg_adverb = pos_counts['ADV'] / n_words if n_words else 0
    avg_punc = sum(1 for char in sentence if char in string.punctuation) / n_words if n_words else 0
    avg_word_length = sum(len(word) for word in tokens) / n_words if n_words else 0

    feature_dict = {
        'n_words': n_words,
        # 'n_bpe_chars': n_bpe_chars,
        # 'avg_bpe': avg_bpe,
        'n_tokens': n_tokens,
        'avg_noun': avg_noun,
        'avg_verb': avg_verb,
        'avg_adj': avg_adj,
        'avg_sat_adj': avg_sat_adj,
        'avg_adverb': avg_adverb,
        'avg_punc': avg_punc,
        'avg_word_length': avg_word_length
    }

    # Add BoW features
    for i, val in enumerate(bow):
        feature_dict[f'bow_{i}'] = val

    return feature_dict


# base_dir = "data/wmt16_de_en"
# base_dir = "../../../wmt16de"
base_dir = "/home/drew99/School/MLDS2/wmt16de"


# Load Spacy's model for part-of-speech tagging
nlp = spacy.load('en_core_web_sm')

# Initialize BPE tokenizer
# bpe_tokenizer = ByteLevelBPETokenizer('path/to/vocab', 'path/to/merges')
# bpe_tokenizer = ByteLevelBPETokenizer(
#     f'{base_dir}/vocab.bpe.32000', 
#     f'{base_dir}/bpe.32000'
# )

# Update paths to where the training files are located
english_file_path = f'{base_dir}/train.en'
german_file_path = f'{base_dir}/train.de'
# # Paths to the tokenized and cleaned training files
# english_file_path = f'{base_dir}/train.tok.clean.en'
# german_file_path = f'{base_dir}/train.tok.clean.de'

# Read the dataset
with open(english_file_path, 'r', encoding='utf-8') as file_en:
    english_sentences = file_en.readlines()

english_sentences = english_sentences[:5000]
# Create a BoW model based on the English sentences
bow_vectorizer = create_bow_model(english_sentences)

# Extract features from English sentences
features_list = [extract_features(sentence, bow_vectorizer) for sentence in english_sentences]

# Convert to DataFrame and save to CSV
df = pd.DataFrame(features_list)
df.to_csv('wmt16_features.csv', index=False)

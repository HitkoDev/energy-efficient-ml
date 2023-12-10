import pandas as pd
import spacy
from nltk.corpus import wordnet as wn
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

def is_satellite_adjective(word):
    synsets = wn.synsets(word, pos=wn.ADJ_SAT)
    return bool(synsets)

def extract_features(sentence, bow_vectorizer, bpe_tokenizer):
    doc = nlp(sentence)
    # Filtered tokens list: includes only words
    word_tokens = [token.text for token in doc if token.is_alpha]
    # Original tokens list: includes words, punctuation, and symbols
    tokens = [token.text for token in doc]
    # Part-of-speech counts
    pos_counts = Counter(token.pos_ for token in doc if token.is_alpha)

    # n_words - count of actual words in the sentence
    # n_tokens - tokens can include words, punctuation, symbols, and other elements
    n_words = len(word_tokens)
    n_tokens = len(word_tokenize(sentence))
    avg_noun = pos_counts['NOUN'] / n_words if n_words else 0
    avg_verb = pos_counts['VERB'] / n_words if n_words else 0
    avg_adj = pos_counts['ADJ'] / n_words if n_words else 0
    # satellite_adjs - https://stackoverflow.com/a/18817481
    satellite_adjs = sum(1 for word in word_tokens if is_satellite_adjective(word))
    avg_sat_adj = satellite_adjs / n_words if n_words else 0
    avg_adverb = pos_counts['ADV'] / n_words if n_words else 0
    avg_punc = sum(1 for char in sentence if char in string.punctuation) / n_words if n_words else 0
    avg_word_length = sum(len(word) for word in word_tokens) / n_words if n_words else 0

    bpe_tokens = bpe_tokenizer.encode(sentence)
    n_bpe_chars = len(bpe_tokens.tokens)
    avg_bpe = n_bpe_chars / n_words if n_words else 0
    
    feature_dict = {
        'n_words': n_words,
        'n_bpe_chars': n_bpe_chars,
        'avg_bpe': avg_bpe,
        'n_tokens': n_tokens,
        'avg_noun': avg_noun,
        'avg_verb': avg_verb,
        'avg_adj': avg_adj,
        'avg_sat_adj': avg_sat_adj,
        'avg_adverb': avg_adverb,
        'avg_punc': avg_punc,
        'avg_word_length': avg_word_length
    }

    bow = bow_vectorizer.transform([sentence]).toarray()[0]
    # Add BoW features
    for i, val in enumerate(bow):
        feature_dict[f'bow_{i}'] = val

    return feature_dict


base_dir = "/home/drew99/School/MLDS2/wmt16de"

# Load Spacy's model for part-of-speech tagging
nlp = spacy.load('en_core_web_sm')

# Ensure WordNet and NLTK's punkt are downloaded
# import nltk
# nltk.download('punkt')
# nltk.download('wordnet')


import json

# Path to the BPE vocabulary file
vocab_file_path = f'{base_dir}/vocab.bpe.32000'
# Path to save the JSON-formatted vocabulary file
json_vocab_file_path = f'{base_dir}/vocab.json'

def convert_vocab_to_json(vocab_file_path, json_vocab_file_path):
    with open(vocab_file_path, 'r', encoding='utf-8') as file:
        vocab = {line.strip(): idx for idx, line in enumerate(file)}

    with open(json_vocab_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(vocab, json_file, ensure_ascii=False)

convert_vocab_to_json(vocab_file_path, json_vocab_file_path)


# Initialize BPE tokenizer
# bpe_tokenizer = ByteLevelBPETokenizer('path/to/vocab', 'path/to/merges')
bpe_tokenizer = ByteLevelBPETokenizer(
    f'{base_dir}/vocab.json', 
    f'{base_dir}/bpe.32000'
)

# Update paths to where the training files are located
english_file_path = f'{base_dir}/train.en'
german_file_path = f'{base_dir}/train.de'
# # Paths to the tokenized and cleaned training files
# english_file_path = f'{base_dir}/train.tok.clean.en'
# german_file_path = f'{base_dir}/train.tok.clean.de'

# Load the first 5000 sentences from the English file
num_sentences = 5000
english_sentences = []
with open(english_file_path, 'r', encoding='utf-8') as file_en:
    for i, line in enumerate(file_en):
        if i < num_sentences:
            english_sentences.append(line)
        else:
            break

# english_sentences = english_sentences[:5000]
# Create a BoW model based on the English sentences
bow_vectorizer = create_bow_model(english_sentences)

# Extract features from English sentences
features_list = [extract_features(sentence, bow_vectorizer, bpe_tokenizer) for sentence in english_sentences]

# Convert to DataFrame and save to CSV
df = pd.DataFrame(features_list)
df.to_csv('wmt16_features.csv', index=False)

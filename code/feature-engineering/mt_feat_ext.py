import pandas as pd
import spacy
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import string

# python -m spacy download en_core_web_sm
import nltk
nltk.download('punkt_tab')
nltk.download('wordnet')

def create_bow_model(sentences):
    vectorizer = CountVectorizer()
    vectorizer.fit(sentences)
    return vectorizer

def is_satellite_adjective(word):
    synsets = wn.synsets(word, pos=wn.ADJ_SAT)
    return bool(synsets)

def extract_features(sentence, bpe_sentence, bow_vectorizer):
    doc = nlp(sentence)
    word_tokens = [token.text for token in doc if token.is_alpha]
    tokens = [token.text for token in doc]
    pos_counts = Counter(token.pos_ for token in doc if token.is_alpha)

    n_words = len(word_tokens)
    n_tokens = len(word_tokenize(sentence))
    avg_noun = pos_counts['NOUN'] / n_words if n_words else 0
    avg_verb = pos_counts['VERB'] / n_words if n_words else 0
    avg_adj = pos_counts['ADJ'] / n_words if n_words else 0
    satellite_adjs = sum(1 for word in word_tokens if is_satellite_adjective(word))
    avg_sat_adj = satellite_adjs / n_words if n_words else 0
    avg_adverb = pos_counts['ADV'] / n_words if n_words else 0
    avg_punc = sum(1 for char in sentence if char in string.punctuation) / n_words if n_words else 0
    avg_word_length = sum(len(word) for word in word_tokens) / n_words if n_words else 0

    # BPE tokenization from the bpe file
    bpe_tokens = bpe_sentence.split()
    n_bpe_chars = len(bpe_tokens)
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
    for i, val in enumerate(bow):
        feature_dict[f'bow_{i}'] = val

    return feature_dict

# Load Spacy's model for part-of-speech tagging
nlp = spacy.load('en_core_web_sm')

base_dir = "/home/drew99/School/MLDS2/wmt16de"

# Loading sentences from files
english_file_path = f'{base_dir}/train.en'
bpe_file_path = f'{base_dir}/train.tok.bpe.32000.en'
print("loading sentences...")
# Load sentences
with open(english_file_path, 'r', newline='\n', encoding='utf-8') as file_en, open(bpe_file_path, 'r', encoding='utf-8') as file_bpe:
    english_sentences = [line.strip() for line in file_en]
    bpe_sentences = [line.strip() for line in file_bpe]

# Ensure the number of sentences in both files are the same
# print(f"Number of sentences in English file: {len(english_sentences)}")
# print(f"Number of sentences in BPE file: {len(bpe_sentences)}")
assert len(english_sentences) == len(bpe_sentences)

# Limit the number of sentences for testing
num_sentences = 5000
english_sentences = english_sentences[:num_sentences]
bpe_sentences = bpe_sentences[:num_sentences]
# print(english_sentences[:5], "\n", bpe_sentences[:5])
print("creating BoW model...")
# Create a BoW model based on the English sentences
bow_vectorizer = create_bow_model(english_sentences)
print("extracting features...")
# Extract features from English sentences
features_list = [extract_features(eng_sent, bpe_sent, bow_vectorizer) for eng_sent, bpe_sent in zip(english_sentences, bpe_sentences)]

# Convert to DataFrame and save to CSV
df = pd.DataFrame(features_list)
df.to_csv('./wmt16_features.csv', index=False)

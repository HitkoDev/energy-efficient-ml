import pandas as pd
import spacy
import nltk
from nltk.tokenize import word_tokenize
from tokenizers import ByteLevelBPETokenizer
from collections import Counter
import string

# Load Spacy's model for part-of-speech tagging
nlp = spacy.load('en_core_web_sm')

# Assuming a pre-trained BPE tokenizer is available
# Replace 'path/to/vocab' and 'path/to/merges' with your actual file paths
bpe_tokenizer = ByteLevelBPETokenizer('path/to/vocab', 'path/to/merges')

def extract_features(sentence):
    # Tokenize and process with spacy for POS tagging
    doc = nlp(sentence)
    tokens = [token.text for token in doc]
    pos_counts = Counter(token.pos_ for token in doc)

    # BPE tokens
    bpe_tokens = bpe_tokenizer.encode(sentence)

    # Calculate features
    n_words = len(tokens)
    n_bpe_chars = len(bpe_tokens.tokens)
    avg_bpe = n_bpe_chars / n_words if n_words else 0
    n_tokens = len(word_tokenize(sentence))
    avg_noun = pos_counts['NOUN'] / n_words if n_words else 0
    avg_verb = pos_counts['VERB'] / n_words if n_words else 0
    avg_adj = pos_counts['ADJ'] / n_words if n_words else 0
    avg_sat_adj = pos_counts['ADJ'] / n_words if n_words else 0  # Assuming satellite adjectives are counted as adjectives
    avg_adverb = pos_counts['ADV'] / n_words if n_words else 0
    avg_punc = sum(1 for char in sentence if char in string.punctuation) / n_words if n_words else 0
    avg_word_length = sum(len(word) for word in tokens) / n_words if n_words else 0

    # BoW (Bag of Words) - Implement based on your specific vocabulary
    # bow = ...

    return {
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
        'avg_word_length': avg_word_length,
        # 'bow': bow,  # Uncomment if you implement BoW
    }

# Example sentences
sentences = ["This is an example sentence.", "Here's another one!"]

# Extract features for each sentence
features_list = [extract_features(sentence) for sentence in sentences]

# Convert to DataFrame
df = pd.DataFrame(features_list)

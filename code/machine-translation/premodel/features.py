import os
import string
from collections import Counter

import pandas as pd
import spacy
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import StratifiedKFold


def create_bow_model(sentences, best):
    vectorizer = CountVectorizer()
    tokenized = vectorizer.fit_transform(sentences)
    # chi2 reduction on BoW
    kbest = SelectKBest(score_func=chi2, k=1500)
    kbest.fit_transform(tokenized, best)
    return vectorizer, kbest


def is_satellite_adjective(word):
    synsets = wn.synsets(word, pos=wn.ADJ_SAT)
    return bool(synsets)


def extract_features(sentence, bpe_sentence, bow_vectorizer, kbest):
    doc = nlp(sentence)
    word_tokens = [token.text for token in doc if token.is_alpha]
    tokens = [token.text for token in doc]
    pos_counts = Counter(token.pos_ for token in doc if token.is_alpha)

    n_words = len(word_tokens)
    n_tokens = len(word_tokenize(sentence))
    avg_noun = pos_counts["NOUN"] / n_words if n_words else 0
    avg_verb = pos_counts["VERB"] / n_words if n_words else 0
    avg_adj = pos_counts["ADJ"] / n_words if n_words else 0
    satellite_adjs = sum(1 for word in word_tokens if is_satellite_adjective(word))
    avg_sat_adj = satellite_adjs / n_words if n_words else 0
    avg_adverb = pos_counts["ADV"] / n_words if n_words else 0
    avg_punc = (
        sum(1 for char in sentence if char in string.punctuation) / n_words
        if n_words
        else 0
    )
    avg_word_length = sum(len(word) for word in word_tokens) / n_words if n_words else 0

    # BPE tokenization from the bpe file
    bpe_tokens = bpe_sentence.split()
    n_bpe_chars = len(bpe_tokens)
    avg_bpe = n_bpe_chars / n_words if n_words else 0

    feature_dict = {
        "n_words": n_words,
        "n_bpe_chars": n_bpe_chars,
        "avg_bpe": avg_bpe,
        "n_tokens": n_tokens,
        "avg_noun": avg_noun,
        "avg_verb": avg_verb,
        "avg_adj": avg_adj,
        "avg_sat_adj": avg_sat_adj,
        "avg_adverb": avg_adverb,
        "avg_punc": avg_punc,
        "avg_word_length": avg_word_length,
    }

    bow = bow_vectorizer.transform([sentence])
    bow = kbest.transform(bow).toarray()[0]
    for i, val in enumerate(bow):
        feature_dict[f"bow_{i}"] = val

    return feature_dict


# Load Spacy's model for part-of-speech tagging
nlp = spacy.load("en_core_web_sm")

base_dir = f"{os.path.dirname(__file__)}/../gnmt/wmt16_de_en"

file = f"{base_dir}/premodels.tok.en"

# Loading sentences from files
tok_file_path = file
bpe_file_path = f"{file[:-3]}.bpe.32000.en"
meta_file = (
    f"{os.path.dirname(__file__)}/../translated/{os.path.basename(file)[:-3]}.csv"
)
target_file = f"{os.path.dirname(__file__)}/data/{os.path.basename(file)}"
os.makedirs(os.path.dirname(target_file), exist_ok=True)
meta = pd.read_csv(meta_file)
best = meta["best_model"].to_list()

# Load sentences
with open(tok_file_path, "r", newline="\n", encoding="utf-8") as file_tok, open(
    bpe_file_path, "r", encoding="utf-8"
) as file_bpe:
    tok_sentences = [line.strip() for line in file_tok]
    bpe_sentences = [line.strip() for line in file_bpe]

assert len(tok_sentences) == len(bpe_sentences)

data = list(zip(tok_sentences, bpe_sentences, best))

kf = StratifiedKFold(n_splits=10)
i = 0
for train, test in kf.split(data, best):
    tok_sentences, bpe_sentences, best = zip(*[data[i] for i in train])
    test_tok_sentences, test_bpe_sentences, test_best = zip(*[data[i] for i in test])

    # Create a BoW model based on the original sentences from the train dataset
    bow_vectorizer, kbest = create_bow_model(tok_sentences, best)

    # Extract features from train sentences
    features_list = [
        extract_features(tok_sent, bpe_sent, bow_vectorizer, kbest)
        for tok_sent, bpe_sent in zip(tok_sentences, bpe_sentences)
    ]

    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(features_list)
    df["sentence"] = train
    df["best_model"] = best
    df.to_csv(f"{target_file}_split_{i}_train.csv", index=False)

    # Extract features from test sentences using trained BoW model
    features_list = [
        extract_features(tok_sent, bpe_sent, bow_vectorizer, kbest)
        for tok_sent, bpe_sent in zip(test_tok_sentences, test_bpe_sentences)
    ]

    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(features_list)
    df["sentence"] = test
    df["best_model"] = test_best
    df.to_csv(f"{target_file}_split_{i}_test.csv", index=False)
    i += 1

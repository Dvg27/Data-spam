"""
Feature Extraction Module for Spam SMS Text Classifier
Extracts hand-crafted numerical, statistical, and structural features from SMS messages.
Handles empty inputs, extreme values, and division by zero safely.
"""

import re
import string
import pandas as pd
import numpy as np


# Currency symbols to track
CURRENCY_REGEX = re.compile(r'[\$£€₹]')
URL_REGEX = re.compile(r'https?://\S+|www\.\S+|\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b|\bbit\.ly\S*|\bgoo\.gl\S*')
REPEATED_CHAR_REGEX = re.compile(r'(.)\1{2,}')


def extract_features_single(message: str) -> dict:
    """
    Extracts 14 numerical features from a single raw SMS message string.
    Safely handles empty messages and division-by-zero.
    """
    if not isinstance(message, str):
        message = "" if message is None else str(message)

    msg_len = len(message)
    words = message.split()
    word_cnt = len(words)

    # Sentence count: split by sentence terminators (. ! ?)
    sentences = [s for s in re.split(r'[.!?]+', message) if s.strip()]
    sentence_cnt = len(sentences) if len(sentences) > 0 else (1 if msg_len > 0 else 0)

    # Punctuation characters
    punct_cnt = sum(1 for c in message if c in string.punctuation)
    punct_ratio = (punct_cnt / msg_len) if msg_len > 0 else 0.0

    # Alphabetic and Uppercase
    alpha_cnt = sum(1 for c in message if c.isalpha())
    upper_cnt = sum(1 for c in message if c.isupper())
    cap_ratio = (upper_cnt / alpha_cnt) if alpha_cnt > 0 else 0.0

    # Digits
    digit_cnt = sum(1 for c in message if c.isdigit())
    digit_ratio = (digit_cnt / msg_len) if msg_len > 0 else 0.0

    # URL presence
    has_url = 1.0 if URL_REGEX.search(message) else 0.0

    # Currency symbols ($ £ € ₹)
    currency_cnt = len(CURRENCY_REGEX.findall(message))

    # Exclamations and Questions
    exclamation_cnt = message.count('!')
    question_cnt = message.count('?')

    # Repeated characters (e.g., 'freeee', '!!!!')
    repeated_chars = len(REPEATED_CHAR_REGEX.findall(message))

    return {
        "message_length": float(msg_len),
        "word_count": float(word_cnt),
        "sentence_count": float(sentence_cnt),
        "punctuation_count": float(punct_cnt),
        "punctuation_ratio": float(punct_ratio),
        "uppercase_count": float(upper_cnt),
        "capitalization_ratio": float(cap_ratio),
        "digit_count": float(digit_cnt),
        "digit_ratio": float(digit_ratio),
        "contains_url": float(has_url),
        "currency_count": float(currency_cnt),
        "exclamation_count": float(exclamation_cnt),
        "question_count": float(question_cnt),
        "repeated_char_count": float(repeated_chars),
    }


def extract_features_batch(messages) -> pd.DataFrame:
    """
    Extracts numerical features for an iterable/Series of raw text messages.
    Returns a pandas DataFrame with normalized column names.
    """
    records = [extract_features_single(msg) for msg in messages]
    df_features = pd.DataFrame(records)
    return df_features


NUMERICAL_FEATURE_NAMES = [
    "message_length",
    "word_count",
    "sentence_count",
    "punctuation_count",
    "punctuation_ratio",
    "uppercase_count",
    "capitalization_ratio",
    "digit_count",
    "digit_ratio",
    "contains_url",
    "currency_count",
    "exclamation_count",
    "question_count",
    "repeated_char_count"
]

"""
Prediction Module for Spam SMS Text Classifier.
Provides inference utilities, single message prediction, probability estimation,
and feature extraction for both the Streamlit web app and CLI usage.
"""

import os
import sys

# Support direct script execution as well as package imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from src.preprocessing import clean_text_for_nlp
from src.feature_extraction import extract_features_single, extract_features_batch, NUMERICAL_FEATURE_NAMES


_CACHED_MODEL = None
_CACHED_VECTORIZER = None
_CACHED_METADATA = None


def load_model_artifacts(models_dir="models"):
    """
    Loads saved model, vectorizer, and metadata artifacts with in-memory caching.
    """
    global _CACHED_MODEL, _CACHED_VECTORIZER, _CACHED_METADATA

    if _CACHED_MODEL is not None and _CACHED_VECTORIZER is not None and _CACHED_METADATA is not None:
        return _CACHED_MODEL, _CACHED_VECTORIZER, _CACHED_METADATA

    if not os.path.isabs(models_dir) and not os.path.exists(models_dir):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        alt_path = os.path.join(project_root, models_dir)
        if os.path.exists(alt_path):
            models_dir = alt_path

    model_path = os.path.join(models_dir, "spam_classifier.pkl")
    vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    metadata_path = os.path.join(models_dir, "feature_metadata.pkl")

    if not (os.path.exists(model_path) and os.path.exists(vectorizer_path) and os.path.exists(metadata_path)):
        raise FileNotFoundError(
            f"Trained model artifacts not found in '{models_dir}'. "
            f"Please run 'python src/train_model.py' to generate model artifacts first."
        )

    _CACHED_MODEL = joblib.load(model_path)
    _CACHED_VECTORIZER = joblib.load(vectorizer_path)
    _CACHED_METADATA = joblib.load(metadata_path)

    return _CACHED_MODEL, _CACHED_VECTORIZER, _CACHED_METADATA


def preprocess_text(text: str) -> str:
    """Public interface for cleaning text."""
    return clean_text_for_nlp(text)


def extract_numerical_features(text: str) -> dict:
    """Public interface for extracting numerical features from a single message."""
    return extract_features_single(text)


def predict_sms(message: str, models_dir="models") -> dict:
    """
    Analyzes an incoming SMS message:
    1. Validates input.
    2. Extracts 14 numerical features.
    3. Cleans text and applies TF-IDF vectorization.
    4. Scales numerical features using the fitted MinMaxScaler.
    5. Combines features and feeds into Naive Bayes classifier.
    6. Returns structured prediction dictionary.
    """
    if not isinstance(message, str) or not message.strip():
        # Handle empty/whitespace input cleanly
        return {
            "error": "Empty message provided. Please enter a valid SMS message.",
            "label": "UNKNOWN",
            "prediction": -1,
            "spam_probability": 0.0,
            "ham_probability": 0.0,
            "features": extract_features_single(""),
            "cleaned_text": ""
        }

    model, vectorizer, metadata = load_model_artifacts(models_dir)
    scaler = metadata.get("scaler")

    # 1. Extract numerical features
    num_features_dict = extract_features_single(message)
    num_df = pd.DataFrame([num_features_dict])[NUMERICAL_FEATURE_NAMES]
    num_vector_scaled = scaler.transform(num_df)

    # 2. Preprocess text and TF-IDF
    cleaned = clean_text_for_nlp(message)
    tfidf_vector = vectorizer.transform([cleaned])

    # 3. Combine features
    combined_features = hstack([tfidf_vector, num_vector_scaled])

    # 4. Predict
    pred_class = int(model.predict(combined_features)[0])
    probabilities = model.predict_proba(combined_features)[0]

    ham_prob = float(probabilities[0])
    spam_prob = float(probabilities[1])

    # 5. Label & confidence
    label = "SPAM" if pred_class == 1 else "HAM"
    confidence = max(ham_prob, spam_prob)

    # 6. Interpret key indicators
    indicators = []
    if num_features_dict["contains_url"] > 0:
        indicators.append("Contains link or URL")
    if num_features_dict["currency_count"] > 0:
        indicators.append(f"Contains {int(num_features_dict['currency_count'])} currency symbol(s)")
    if num_features_dict["exclamation_count"] >= 2:
        indicators.append(f"Multiple exclamation marks ({int(num_features_dict['exclamation_count'])})")
    if num_features_dict["capitalization_ratio"] > 0.3:
        indicators.append(f"High capitalization ratio ({num_features_dict['capitalization_ratio'] * 100:.1f}%)")
    if num_features_dict["digit_ratio"] > 0.15:
        indicators.append(f"High density of numeric digits ({num_features_dict['digit_ratio'] * 100:.1f}%)")

    return {
        "raw_message": message,
        "cleaned_text": cleaned,
        "prediction": pred_class,
        "label": label,
        "spam_probability": spam_prob,
        "ham_probability": ham_prob,
        "confidence": confidence,
        "features": num_features_dict,
        "indicators": indicators,
        "error": None
    }


if __name__ == "__main__":
    # Quick sanity test on sample spam and ham
    test_samples = [
        "Congratulations! You have won a free prize. Call now to claim your reward!",
        "Hey, are you coming to college tomorrow?",
        "URGENT! You have won £1000. Click the link now to claim your prize.",
        "Please call me when you reach home."
    ]

    print("=== Testing prediction.py ===")
    for s in test_samples:
        res = predict_sms(s)
        print(f"Message: {s[:50]}...")
        print(f" -> Label: {res['label']} (Spam: {res['spam_probability']*100:.2f}%, Ham: {res['ham_probability']*100:.2f}%)")
        print(f" -> Indicators: {res['indicators']}")
        print()

"""
System Robustness & Verification Test Suite for Spam SMS Text Classifier.
Tests edge cases, empty strings, extreme lengths, symbols, and evaluates performance metrics.
"""

import os
import sys
import unittest
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import clean_text_for_nlp, map_labels
from src.feature_extraction import extract_features_single, extract_features_batch, NUMERICAL_FEATURE_NAMES
from src.prediction import predict_sms, load_model_artifacts


class TestSpamClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model, cls.vectorizer, cls.metadata = load_model_artifacts("models")

    def test_model_artifacts_loaded(self):
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.vectorizer)
        self.assertIsNotNone(self.metadata)
        self.assertIn("metrics", self.metadata)
        self.assertGreater(self.metadata["metrics"]["accuracy"], 0.95)
        self.assertGreater(self.metadata["metrics"]["spam_precision"], 0.95)

    def test_empty_and_whitespace_input(self):
        res_empty = predict_sms("")
        self.assertIsNotNone(res_empty["error"])
        self.assertEqual(res_empty["label"], "UNKNOWN")

        res_space = predict_sms("     \t \n  ")
        self.assertIsNotNone(res_space["error"])
        self.assertEqual(res_space["label"], "UNKNOWN")

    def test_very_short_messages(self):
        shorts = ["k", "ok", "Hi", "yes", "no"]
        for s in shorts:
            res = predict_sms(s)
            self.assertIsNone(res["error"])
            self.assertIn(res["label"], ["HAM", "SPAM"])
            self.assertEqual(res["label"], "HAM", f"Short message '{s}' should be classified as HAM")

    def test_promotional_spam_messages(self):
        spams = [
            "Congratulations! You have won a $1000 cash prize. Call 08000930705 now to claim!",
            "URGENT! You have won £1000. Click the link http://claim-now.com to claim your prize.",
            "Free entry into our weekly £5000 competition! Text WIN to 80088 immediately.",
            "WINNER!! As a valued customer you have been selected to receive a £900 prize reward. Call now."
        ]
        for s in spams:
            res = predict_sms(s)
            self.assertIsNone(res["error"])
            self.assertEqual(res["label"], "SPAM", f"Expected SPAM for message: {s}")
            self.assertGreater(res["spam_probability"], 0.70)

    def test_normal_conversations(self):
        hams = [
            "Hey, are you coming to college tomorrow?",
            "Please call me when you reach home.",
            "I'm at the library studying for data science exam. See you later!",
            "Can you send me the notes from yesterday's class?",
            "What time are we meeting mom for dinner tonight?"
        ]
        for h in hams:
            res = predict_sms(h)
            self.assertIsNone(res["error"])
            self.assertEqual(res["label"], "HAM", f"Expected HAM for message: {h}")
            self.assertGreater(res["ham_probability"], 0.70)

    def test_extreme_numerical_and_symbolic_signals(self):
        msg = "URGENT $$$$$$ FREE CASH £££ CALL 9999999999 NOW!!!!"
        res = predict_sms(msg)
        self.assertEqual(res["label"], "SPAM")
        self.assertGreaterEqual(res["features"]["currency_count"], 5)
        self.assertGreaterEqual(res["features"]["exclamation_count"], 4)

    def test_very_long_message(self):
        long_ham = "Hello my friend, " + "we should definitely catch up this weekend. " * 15
        res = predict_sms(long_ham)
        self.assertIsNone(res["error"])
        self.assertEqual(res["label"], "HAM")
        self.assertGreater(res["features"]["message_length"], 200)

    def test_numerical_feature_keys(self):
        feats = extract_features_single("Test message with 123 numbers and $10 cost!")
        for col in NUMERICAL_FEATURE_NAMES:
            self.assertIn(col, feats)
            self.assertIsInstance(feats[col], (int, float))

    def test_preprocessing_integrity(self):
        clean = clean_text_for_nlp("Check out https://scam.com and WIN $500 today!!!")
        self.assertNotIn("https", clean)
        self.assertNotIn("scam.com", clean)
        self.assertIsInstance(clean, str)


if __name__ == "__main__":
    unittest.main()

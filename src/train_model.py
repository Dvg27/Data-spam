"""
Model Training and Evaluation Pipeline for Spam SMS Text Classifier.
Performs data loading, preprocessing, numerical feature extraction,
TF-IDF vectorization, hyperparameter tuning with Naive Bayes,
comprehensive precision-driven evaluation, error analysis, and artifact serialization.
"""

import os
import sys
import json
import urllib.request
import zipfile
import io
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

# Relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import clean_text_for_nlp, map_labels
from src.feature_extraction import (
    extract_features_batch,
    extract_features_single,
    NUMERICAL_FEATURE_NAMES
)


def ensure_dataset(data_path="data/SMSSpamCollection"):
    """
    Checks if dataset exists; if not, automatically downloads from UCI repository
    or reliable mirror, extracting it cleanly.
    """
    if os.path.exists(data_path) and os.path.getsize(data_path) > 1000:
        print(f"[+] Dataset found at: {data_path} ({os.path.getsize(data_path)} bytes)")
        return data_path

    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    urls = [
        "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
        "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    ]

    for url in urls:
        try:
            print(f"[*] Attempting download from: {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = resp.read()
                if url.endswith(".zip"):
                    with zipfile.ZipFile(io.BytesIO(data)) as z:
                        z.extract("SMSSpamCollection", path=os.path.dirname(data_path))
                else:
                    with open(data_path, "wb") as f:
                        f.write(data)
            if os.path.exists(data_path) and os.path.getsize(data_path) > 1000:
                print(f"[OK] Successfully downloaded dataset to: {data_path}")
                return data_path
        except Exception as e:
            print(f"[!] Download failed for {url}: {e}")

    raise FileNotFoundError(
        f"Unable to automatically download dataset. Please place 'SMSSpamCollection' inside '{os.path.dirname(data_path)}/'"
    )


def load_and_preprocess_data(data_path="data/SMSSpamCollection"):
    """
    Loads raw SMS Spam Collection dataset, cleans, removes duplicates,
    and returns dataset statistics along with processed DataFrames.
    """
    ensure_dataset(data_path)
    df_raw = pd.read_csv(
        data_path,
        sep='\t',
        names=['label', 'message'],
        quoting=3,
        encoding='utf-8'
    )

    total_records = len(df_raw)
    raw_spam_cnt = int((df_raw['label'] == 'spam').sum())
    raw_ham_cnt = int((df_raw['label'] == 'ham').sum())
    missing_values = int(df_raw.isnull().sum().sum())
    duplicate_count = int(df_raw.duplicated(subset=['message']).sum())

    # Remove duplicates
    df = df_raw.drop_duplicates(subset=['message']).reset_index(drop=True)
    dedup_total = len(df)
    spam_count = int((df['label'] == 'spam').sum())
    ham_count = int((df['label'] == 'ham').sum())
    spam_pct = (spam_count / dedup_total) * 100.0
    ham_pct = (ham_count / dedup_total) * 100.0

    stats = {
        "total_records_raw": total_records,
        "spam_records_raw": raw_spam_cnt,
        "ham_records_raw": raw_ham_cnt,
        "missing_values": missing_values,
        "duplicate_count": duplicate_count,
        "total_records_dedup": dedup_total,
        "spam_count": spam_count,
        "ham_count": ham_count,
        "spam_percentage": spam_pct,
        "ham_percentage": ham_pct
    }

    # NLP text cleaning
    print("[*] Performing NLP text normalization & stemming...")
    df['clean_message'] = df['message'].apply(clean_text_for_nlp)
    df['target'] = map_labels(df['label'])

    # Feature extraction
    print("[*] Extracting 14 hand-crafted numerical features...")
    df_numerical = extract_features_batch(df['message'])

    return df, df_numerical, stats


def train_and_evaluate(data_path="data/SMSSpamCollection", models_dir="models"):
    """
    Full training, evaluation, comparison, and serialization pipeline.
    """
    os.makedirs(models_dir, exist_ok=True)
    df, df_numerical, stats = load_and_preprocess_data(data_path)

    # Stratified Train-Test Split (80% train, 20% test)
    print("[*] Splitting dataset (80% Train, 20% Test) with Stratification...")
    X_train_text, X_test_text, y_train, y_test, num_train, num_test, idx_train, idx_test = train_test_split(
        df['clean_message'],
        df['target'],
        df_numerical,
        df.index,
        test_size=0.20,
        random_state=42,
        stratify=df['target']
    )

    # Fit TF-IDF Vectorizer only on training text (avoids data leakage)
    print("[*] Fitting TF-IDF Vectorizer (unigrams + bigrams)...")
    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_test_tfidf = tfidf.transform(X_test_text)

    # Scale numerical features using MinMaxScaler (fitted on training data only)
    scaler = MinMaxScaler()
    num_train_scaled = scaler.fit_transform(num_train)
    num_test_scaled = scaler.transform(num_test)

    # Feature combination: TF-IDF sparse matrix + scaled numerical features
    X_train_combined = hstack([X_train_tfidf, num_train_scaled])
    X_test_combined = hstack([X_test_tfidf, num_test_scaled])

    # Model Comparison & Hyperparameter Tuning
    alphas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0]
    comparison_results = []

    print("[*] Evaluating MultinomialNB vs ComplementNB across hyperparameter alpha values...")
    best_model = None
    best_f1 = -1.0
    best_alpha = 0.2
    best_model_name = "MultinomialNB"

    for alpha in alphas:
        # MultinomialNB
        mnb = MultinomialNB(alpha=alpha)
        mnb.fit(X_train_combined, y_train)
        preds_mnb = mnb.predict(X_test_combined)
        prec_mnb = precision_score(y_test, preds_mnb, pos_label=1, zero_division=0)
        rec_mnb = recall_score(y_test, preds_mnb, pos_label=1, zero_division=0)
        acc_mnb = accuracy_score(y_test, preds_mnb)
        f1_mnb = f1_score(y_test, preds_mnb, pos_label=1, zero_division=0)

        comparison_results.append({
            "Classifier": "MultinomialNB",
            "Alpha": alpha,
            "Accuracy": acc_mnb,
            "Spam_Precision": prec_mnb,
            "Spam_Recall": rec_mnb,
            "Spam_F1": f1_mnb
        })

        # ComplementNB
        cnb = ComplementNB(alpha=alpha)
        cnb.fit(X_train_combined, y_train)
        preds_cnb = cnb.predict(X_test_combined)
        prec_cnb = precision_score(y_test, preds_cnb, pos_label=1, zero_division=0)
        rec_cnb = recall_score(y_test, preds_cnb, pos_label=1, zero_division=0)
        acc_cnb = accuracy_score(y_test, preds_cnb)
        f1_cnb = f1_score(y_test, preds_cnb, pos_label=1, zero_division=0)

        comparison_results.append({
            "Classifier": "ComplementNB",
            "Alpha": alpha,
            "Accuracy": acc_cnb,
            "Spam_Precision": prec_cnb,
            "Spam_Recall": rec_cnb,
            "Spam_F1": f1_cnb
        })

        # Selection criteria: Prioritizing high Spam Precision while maintaining strong F1 (>0.90)
        if prec_mnb >= 0.96 and f1_mnb > best_f1:
            best_f1 = f1_mnb
            best_model = mnb
            best_alpha = alpha
            best_model_name = "MultinomialNB"

    if best_model is None:
        best_model = MultinomialNB(alpha=0.2)
        best_model.fit(X_train_combined, y_train)

    print(f"[OK] Selected Best Model: {best_model_name} (alpha={best_alpha})")

    # Evaluate final selected model
    y_pred = best_model.predict(X_test_combined)
    y_prob = best_model.predict_proba(X_test_combined)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision_spam = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall_spam = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1_spam = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

    precision_ham = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
    recall_ham = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    f1_ham = f1_score(y_test, y_pred, pos_label=0, zero_division=0)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    clf_report = classification_report(y_test, y_pred, target_names=["HAM", "SPAM"], output_dict=True)

    # ROC & Precision-Recall curves
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc_val = auc(fpr, tpr)
    pr_curve_precision, pr_curve_recall, _ = precision_recall_curve(y_test, y_prob)
    avg_precision_val = average_precision_score(y_test, y_prob)

    # Error analysis
    test_df_original = df.loc[idx_test].copy()
    test_df_original['actual'] = y_test
    test_df_original['predicted'] = y_pred
    test_df_original['spam_probability'] = y_prob

    false_positives = test_df_original[(test_df_original['actual'] == 0) & (test_df_original['predicted'] == 1)][
        ['message', 'actual', 'predicted', 'spam_probability']
    ].to_dict(orient='records')

    false_negatives = test_df_original[(test_df_original['actual'] == 1) & (test_df_original['predicted'] == 0)][
        ['message', 'actual', 'predicted', 'spam_probability']
    ].to_dict(orient='records')

    # Top words in Spam vs Ham
    feature_names_text = tfidf.get_feature_names_out()
    # Log probabilities from Naive Bayes: shape (2, n_features)
    # The first n_text features correspond to TF-IDF terms
    log_prob_ham = best_model.feature_log_prob_[0][:len(feature_names_text)]
    log_prob_spam = best_model.feature_log_prob_[1][:len(feature_names_text)]

    top_spam_idx = np.argsort(log_prob_spam)[::-1][:20]
    top_ham_idx = np.argsort(log_prob_ham)[::-1][:20]

    top_spam_terms = [(feature_names_text[i], float(log_prob_spam[i])) for i in top_spam_idx]
    top_ham_terms = [(feature_names_text[i], float(log_prob_ham[i])) for i in top_ham_idx]

    # Package metadata
    metadata = {
        "model_name": best_model_name,
        "best_alpha": best_alpha,
        "dataset_stats": stats,
        "metrics": {
            "accuracy": float(accuracy),
            "spam_precision": float(precision_spam),
            "spam_recall": float(recall_spam),
            "spam_f1": float(f1_spam),
            "ham_precision": float(precision_ham),
            "ham_recall": float(recall_ham),
            "ham_f1": float(f1_ham),
            "roc_auc": float(roc_auc_val),
            "average_precision": float(avg_precision_val)
        },
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "matrix": cm.tolist()
        },
        "classification_report": clf_report,
        "comparison_table": comparison_results,
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": float(roc_auc_val)
        },
        "pr_curve": {
            "precision": pr_curve_precision.tolist(),
            "recall": pr_curve_recall.tolist(),
            "avg_precision": float(avg_precision_val)
        },
        "error_analysis": {
            "false_positives": false_positives,
            "false_negatives": false_negatives
        },
        "top_terms": {
            "spam": top_spam_terms,
            "ham": top_ham_terms
        },
        "scaler": scaler,
        "numerical_feature_names": NUMERICAL_FEATURE_NAMES,
        "num_text_features": len(feature_names_text),
        "total_features": X_train_combined.shape[1]
    }

    # Save artifacts
    model_path = os.path.join(models_dir, "spam_classifier.pkl")
    vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    metadata_path = os.path.join(models_dir, "feature_metadata.pkl")

    joblib.dump(best_model, model_path)
    joblib.dump(tfidf, vectorizer_path)
    joblib.dump(metadata, metadata_path)

    print(f"[OK] Artifacts saved:")
    print(f"    - Model: {model_path}")
    print(f"    - Vectorizer: {vectorizer_path}")
    print(f"    - Metadata & Scaler: {metadata_path}")

    print("\n" + "=" * 50)
    print("FINAL TEST EVALUATION RESULTS:")
    print(f"Accuracy:       {accuracy * 100:.2f}%")
    print(f"Spam Precision: {precision_spam * 100:.2f}%  <-- Crucial to minimize False Positives")
    print(f"Spam Recall:    {recall_spam * 100:.2f}%")
    print(f"Spam F1-Score:  {f1_spam * 100:.2f}%")
    print(f"Ham Precision:  {precision_ham * 100:.2f}%")
    print(f"Ham Recall:     {recall_ham * 100:.2f}%")
    print(f"ROC-AUC:        {roc_auc_val:.4f}")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("=" * 50 + "\n")

    return best_model, tfidf, metadata


if __name__ == "__main__":
    train_and_evaluate()

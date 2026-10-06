"""
NLTK Data Setup Script — Run during Vercel build to pre-download required corpora.
Downloads stopwords and punkt tokenizer into a local `nltk_data/` directory
so they are bundled into the serverless function at deploy time.
"""
import os
import nltk

# Download into a local directory that gets bundled via includeFiles
NLTK_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)

print(f"Downloading NLTK data to: {NLTK_DATA_DIR}")
nltk.download("stopwords", download_dir=NLTK_DATA_DIR, quiet=False)
nltk.download("punkt", download_dir=NLTK_DATA_DIR, quiet=False)
nltk.download("punkt_tab", download_dir=NLTK_DATA_DIR, quiet=False)
print("NLTK data download complete.")

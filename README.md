# 🎬 IMDb Sentiment Analysis

This project performs **sentiment analysis** on the IMDb movie reviews dataset using Python.  
It classifies each review as **Positive, Negative, or Neutral** based on polarity scores.

---

## 📌 Task (TEAS Framework)

- **T (Task):** Analyze IMDb movie reviews and classify them into sentiment categories.  
- **E (Execution):** Used Python (`pandas`, `textblob`, `tqdm`). Processed **25,000 reviews** (half dataset).  
- **A (Analysis):** Each review was assigned a polarity score:
  - `> 0` → Positive  
  - `< 0` → Negative  
  - `= 0` → Neutral  
- **S (Summary):** Majority of reviews are **Positive**, fewer are **Negative**, and a small fraction are **Neutral**.

---

## 📂 Files

- `imdb_sentiment.py` → Python script for analysis  
- `IMDB_sentiment_results.csv` → Output file with reviews + predicted sentiments  
- `IMDB Dataset.csv` → Original dataset (not included here due to size)

---

## ⚡ How to Run

1. Install requirements:
   ```bash
   pip install pandas textblob tqdm
   python -m textblob.download_corpora

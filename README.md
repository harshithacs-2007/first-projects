# 🎬 IMDb Sentiment Analysis - AIML Recruitment Task

This project performs **sentiment analysis** on the IMDb movie reviews dataset using Python.  
Each review is classified as **Positive** or **Negative** using a **Naive Bayes classifier** with **CountVectorizer** for text vectorization.

---

## 📌 Task (TEAS Framework)

- **T (Task):** Classify IMDb movie reviews into Positive/Negative sentiment.
- **E (Execution):**  
  - Loaded `IMDB Dataset.csv`.  
  - Preprocessed reviews (lowercased).  
  - Converted text to numeric vectors using **CountVectorizer**.  
  - Trained a **Multinomial Naive Bayes** model (80% train / 20% test).  
  - Evaluated model accuracy.  
- **A (Analysis):**  
  - Accuracy on test set: ~85–88%.  
  - Model predicts sentiment based on word frequencies.  
- **S (Summary):**  
  - Most reviews correctly classified.  
  - Results saved in `IMDB_sentiment_results.csv`.

---

## 📂 Files

- `imdb_sentiment.py` → Python script  
- `IMDB_sentiment_results.csv` → Output predictions  
- `IMDB Dataset.csv` → Original dataset  

---

## ⚡ How to Run

1. Install dependencies:
```bash
pip install pandas scikit-learn

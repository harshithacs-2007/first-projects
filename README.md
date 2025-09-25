# 🎬 IMDb Sentiment Analysis – Naive Bayes

## 📌 Overview
This project is part of the **AIML Recruitment Task (First Year)**.  
We use the **IMDb Movie Review Dataset** (50,000 reviews: 25k positive, 25k negative) to build a simple **Naive Bayes classifier** that predicts whether a review is **positive** or **negative**.

---

## ⚙️ Approach
1. **Preprocessing**
   - Converted all reviews to lowercase  
   - Removed stopwords using `CountVectorizer`  
   - Limited features to top 5000 words for efficiency  
   - Converted sentiment labels: `positive → 1`, `negative → 0`  

2. **Model**
   - Used **Multinomial Naive Bayes** (`MultinomialNB`)  
   - Train-test split: **80% training, 20% testing**  

3. **Evaluation**
   - Accuracy: ~85–87% (depends on data split)  
   - Classification report includes **precision, recall, F1-score**

---

## 📂 Files
- `sentiment_nb.py` → main Python script  
- `IMDB Dataset.csv` → dataset (50,000 reviews, balanced)  
- `IMDB_sentiment_results.csv` → predictions after running the script  

---

## ▶️ How to Run
```bash
pip install pandas scikit-learn
python sentiment_nb.py

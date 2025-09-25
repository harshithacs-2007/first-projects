# IMDb Sentiment Analysis using Naive Bayes
# Author: <your name>
# AIML Recruitment Task - First Year (2025 Batch)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("IMDB Dataset.csv")  # Ensure file is in same folder

# For speed during training, you can sample data (optional)
df = df.head(25000).copy()

# -----------------------------
# 2. Preprocessing
# -----------------------------
df['review'] = df['review'].str.lower()  # lowercase text

# Convert labels to binary (positive=1, negative=0)
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

X = df['review']
y = df['sentiment']

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 3. Text Vectorization
# -----------------------------
vectorizer = CountVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# 4. Train Model
# -----------------------------
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# -----------------------------
# 5. Evaluation
# -----------------------------
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)

print(f"✅ Model Accuracy: {acc*100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# 6. Save Results
# -----------------------------
df['Predicted_Sentiment'] = model.predict(vectorizer.transform(df['review']))
df.to_csv("IMDB_sentiment_results.csv", index=False)

# Show some sample predictions
print("\nSample predictions:")
print(df[['review', 'Predicted_Sentiment']].head(10))

# IMDb Sentiment Analysis - Naive Bayes
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# 1. Load dataset
df = pd.read_csv("IMDB Dataset.csv")

# 2. Optional: use half data for speed
df_small = df.head(25000).copy()

# 3. Preprocessing: lowercase
df_small['review'] = df_small['review'].str.lower()

# 4. Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    df_small['review'], df_small['sentiment'], test_size=0.2, random_state=42
)

# 5. Convert text to vectors
vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 6. Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 7. Predict on test set
y_pred = model.predict(X_test_vec)

# 8. Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc*100:.2f}%")

# 9. Save predictions for full dataset
X_vec_full = vectorizer.transform(df_small['review'])
df_small['Predicted_Sentiment'] = model.predict(X_vec_full)
df_small.to_csv("IMDB_sentiment_results.csv", index=False)

# 10. Print sample
print("\nSample predictions:")
print(df_small[['review','Predicted_Sentiment']].head(10))

# IMDb Sentiment Analysis - Half Dataset (25k reviews with Progress Bar + Summary)
import pandas as pd
from textblob import TextBlob
from tqdm import tqdm

# 1. Load dataset
df = pd.read_csv("IMDB Dataset.csv")

# 2. Use half of the data (25,000 reviews)
df_half = df.head(25000).copy()

# 3. Function to calculate sentiment
def get_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# 4. Apply with progress bar
tqdm.pandas()
df_half["Predicted_Sentiment"] = df_half["review"].progress_apply(get_sentiment)

# 5. Save results
df_half.to_csv("IMDB_sentiment_results.csv", index=False)

# 6. Print first few results
print("\nSample Results:")
print(df_half[["review", "Predicted_Sentiment"]].head(10))

# 7. Print sentiment counts + percentages
print("\nSummary Report:")
summary_counts = df_half["Predicted_Sentiment"].value_counts()
summary_percent = df_half["Predicted_Sentiment"].value_counts(normalize=True) * 100

for sentiment in summary_counts.index:
    print(f"{sentiment}: {summary_counts[sentiment]} reviews ({summary_percent[sentiment]:.2f}%)")

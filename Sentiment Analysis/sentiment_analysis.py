import pandas as pd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob


# Load the medium-sized English spaCy model.
nlp = spacy.load("en_core_web_md")

# Add SpacyTextBlob to the pipeline to calculate sentiment polarity.
nlp.add_pipe("spacytextblob")


# Load the Amazon product reviews dataset.
data = pd.read_csv(
    "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv"
)


# Remove rows that contain missing review text.
clean_data = data.dropna(subset=["reviews.text"]).copy()

# Select the product review text that will be analysed.
reviews_data = clean_data["reviews.text"]


def preprocess_review(review):
    """Clean a product review before sentiment analysis."""

    # Convert the review to a lowercase string and remove extra spaces.
    review = str(review).lower().strip()

    # Process the review using spaCy.
    doc = nlp(review)

    # Remove stop words, punctuation, and whitespace.
    filtered_tokens = [
        token.text
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
    ]

    return " ".join(filtered_tokens)


def analyse_sentiment(review):
    """Predict the sentiment of a product review."""

    # Preprocess the review before analysing its sentiment.
    cleaned_review = preprocess_review(review)

    # Process the cleaned review with spaCy.
    doc = nlp(cleaned_review)

    # Obtain the polarity score using SpacyTextBlob.
    polarity = doc._.blob.polarity

    # Classify the review according to its polarity score.
    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return sentiment, polarity


# Display basic information about the dataset.
print("Amazon Product Reviews Sentiment Analysis")
print("-" * 45)
print(f"Number of reviews: {len(reviews_data)}")


# Select a few reviews from the dataset to test the model.
sample_reviews = reviews_data.iloc[:5]

print("\nSample Sentiment Analysis Results")
print("-" * 45)

for index, review in enumerate(sample_reviews, start=1):
    sentiment, polarity = analyse_sentiment(review)

    print(f"\nReview {index}:")
    print(review)
    print(f"Predicted sentiment: {sentiment}")
    print(f"Polarity score: {polarity:.4f}")

# Compare the similarity between two product reviews
review_one = reviews_data.iloc[0]
review_two = reviews_data.iloc[1]

doc_one = nlp(str(review_one))
doc_two = nlp(str(review_two))

similarity_score = doc_one.similarity(doc_two)

print("\nReview Similarity Comparison")
print("-" * 45)
print(f"Review 1: {review_one}")
print(f"Review 2: {review_two}")
print(f"Similarity score: {similarity_score:.4f}")
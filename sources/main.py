import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from build_model import SentimentAnalysisModel
from feature_engineering import extract_features
from preprocessing import preprocess_data
from sklearn.model_selection import train_test_split


def main():
    # Specify the relative path to the JSON file from the current directory
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, "..", "sample_data", "reviews.json")

    # Load and preprocess data
    df = pd.read_json(file_path)
    preprocess_data(df)

    # Extract features
    X = extract_features(df)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        df["review_text"], df["sentiment"], test_size=0.2, random_state=42
    )

    # Print and inspect the data
    print("X_train['review_text']:", X_train.tolist())
    print("y_train:", y_train.tolist())

    # Create and train the model
    sentiment_model = SentimentAnalysisModel()
    sentiment_model.train(
        X_train.tolist(), np.array(y_train).tolist()
    )  # Convert y_train to a NumPy array

    # Evaluate the model
    sentiment_model.evaluate(X_test.tolist(), np.array(y_test).tolist())

    # Make predictions on the test set
    test_predictions = sentiment_model.predict(X_test.tolist())

    # Visualize the distribution of sentiment scores on the test set
    visualize_sentiment_distribution(test_predictions)


def visualize_sentiment_distribution(sentiment_scores):
    plt.hist(
        sentiment_scores, bins=20, edgecolor="black"
    )  # Adjust the number of bins as needed
    plt.title("Distribution of Sentiment Scores")
    plt.xlabel("Sentiment Score")
    plt.ylabel("Number of Reviews")
    plt.show()


if __name__ == "__main__":
    main()

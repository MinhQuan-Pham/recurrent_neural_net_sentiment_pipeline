import numpy as np
import pandas as pd
from build_model import SentimentAnalysisModel
from feature_engineering import extract_features
from preprocessing import preprocess_data
from sklearn.model_selection import train_test_split


def main():
    # Load and preprocess data
    df = pd.read_json("reviews_extended.json")
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
    sentiment_model.evaluate(
        X_test.tolist(), np.array(y_test).tolist()
    )  # Convert y_test to a NumPy array

    # Make predictions on new data
    new_data = ["This is a great product!", "Super satisfied with the service."]
    new_predictions = sentiment_model.predict(new_data)

    # Print the sentiment predictions
    print("Sentiment Predictions:", new_predictions)


if __name__ == "__main__":
    main()

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Dense

class SentimentAnalysisModel:
    def __init__(self, max_words=5000, embedding_dim=128, lstm_units=100):
        self.max_words = max_words
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.tokenizer = Tokenizer(num_words=self.max_words)
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=self.max_words, output_dim=self.embedding_dim, input_length=self.max_words))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(self.lstm_units))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model

    def train(self, X_train, y_train, epochs=5, batch_size=64):
        self.tokenizer.fit_on_texts(X_train)
        X_train_sequences = self.tokenizer.texts_to_sequences(X_train)
        X_train_padded = pad_sequences(X_train_sequences, maxlen=self.max_words)
        y_train = np.array(y_train)  # Convert y_train to a NumPy array
        self.model.fit(X_train_padded, y_train, epochs=epochs, batch_size=batch_size)


    def evaluate(self, X_test, y_test):
        X_test_sequences = self.tokenizer.texts_to_sequences(X_test)
        X_test_padded = pad_sequences(X_test_sequences, maxlen=self.max_words)
        y_test = np.array(y_test)  # Convert y_test to a NumPy array
        loss, accuracy = self.model.evaluate(X_test_padded, y_test)
        print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")


    def predict(self, new_data):
        new_data_sequences = self.tokenizer.texts_to_sequences(new_data)
        new_data_padded = pad_sequences(new_data_sequences, maxlen=self.max_words)
        predictions = self.model.predict(new_data_padded)
        return predictions


# Example Usage:
if __name__ == "__main__":
    # Instantiate the model
    sentiment_model = SentimentAnalysisModel()

    # Example training data and labels
    X_train = [
        "This is a positive review.",
        "Negative sentiment in this one.",
        "Another positive example.",
        "Not happy with this product.",
    ]
    y_train = np.array([1, 0, 1, 0])

    # Train the model
    sentiment_model.train(X_train, y_train)

    # Example testing data and labels
    X_test = [
        "Another positive example.",
        "Not happy with this product.",
    ]
    y_test = np.array([1, 0])

    # Evaluate the model
    sentiment_model.evaluate(X_test, y_test)

    # Example prediction on new data
    new_data = ["I love this product!", "Disappointed with the service."]
    predictions = sentiment_model.predict(new_data)

    # Print the sentiment predictions
    print("Sentiment Predictions:", predictions)

# Import necessary libraries
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


def extract_features(df):
    tokenizer = Tokenizer(num_words=5000)  # Adjust num_words based on your dataset
    tokenizer.fit_on_texts(df['review_text'])
    X = tokenizer.texts_to_sequences(df['review_text'])
    X = pad_sequences(X)

    return X
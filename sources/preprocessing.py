from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')


def preprocess_data(data):
    # Drop rows with missing values
    data = data.dropna()

    # Lowercase conversion
    data["review_text"] = data["review_text"].apply(lambda x: x.lower())

    # Tokenization
    data["tokenized_text"] = data["review_text"].apply(tokenize_text)

    # Stopword removal
    data["tokenized_text"] = data["tokenized_text"].apply(remove_stopwords)

    # Lemmatization
    data["tokenized_text"] = data["tokenized_text"].apply(lemmatize_text)

    # Additional text cleaning steps if needed

    return data


def tokenize_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    return tokens


def remove_stopwords(tokens):
    # Remove stop words (including common English words)
    stop_words = set(stopwords.words("english") + list(ENGLISH_STOP_WORDS))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens


def lemmatize_text(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens


## Additional preprocessing steps if needed

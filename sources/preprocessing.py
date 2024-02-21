# preprocessing.py
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

"""
Logic Explanation:
e.g. "The quick brown fox jumped over the lazy dog."

1. Tokenization: 
- Break the string into individual words or "tokens"
- Return: ['The', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog', '.']

2. Removal of Stop Words: 
- Stop words are common words that are often removed from text because they are considered to carry little meaning.
- Return: ['quick', 'brown', 'fox', 'jumped', 'lazy', 'dog', '.']

3. Lemmatization
- Lemmatization reduces words to their base or root form. For example, "jumped" becomes "jump," and "dogs" becomes "dog."
- Return: ['quick', 'brown', 'fox', 'jumped', 'lazy', 'dog', '.']

"""

def preprocess_data(data):
    # Drop rows with missing values
    data = data.dropna()

    # Lowercase conversion
    data['review_text'] = data['review_text'].apply(lambda x: x.lower())

    # Tokenization
    data['tokenized_text'] = data['review_text'].apply(tokenize_text)

    # Stopword removal
    data['tokenized_text'] = data['tokenized_text'].apply(remove_stopwords)

    # Lemmatization
    data['tokenized_text'] = data['tokenized_text'].apply(lemmatize_text)

    # Additional text cleaning steps if needed

    return data

def tokenize_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    return tokens

def remove_stopwords(tokens):
    # Remove stop words (including common English words)
    stop_words = set(stopwords.words('english') + list(ENGLISH_STOP_WORDS))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens

def lemmatize_text(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

## Additional preprocessing steps if needed
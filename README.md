# Recurrent Neural Network Sentiment Analysis Pipeline

## Overview

This repository contains a sentiment analysis pipeline designed to analyze and classify textual reviews as positive or negative. The pipeline includes modules for data preprocessing, feature engineering, model training, and prediction. Additionally, it provides functionality to visualize the distribution of sentiment scores.

![image](https://github.com/MinhQuan-Pham/customer_reviews_nlp_decoder/assets/65067055/bca6a022-b8bf-4609-bebc-08827636bab2)

## Model Architecture Design

The neural network architecture was chosen based on the nature of sentiment analysis tasks. Here's a breakdown of key components:

- **Embedding Layer:**
  - Converts words into vectors and represents the semantic relationships between them.
  - Helps the model learn meaningful representations of words in the given context.

- **Spatial Dropout:**
  - Regularizes the model by randomly dropping entire 1D feature maps during training.
  - Prevents overfitting by adding noise to the network.

- **LSTM Layer:**
  - Long Short-Term Memory (LSTM) networks are well-suited for sequence data.
  - Captures long-range dependencies in the input sequences.
  - Effective in understanding the context and sentiment in sentences.

- **Dense Layer with Sigmoid Activation:**
  - Produces a binary output, indicating positive or negative sentiment.
  - Sigmoid activation is suitable for binary classification problems.

The chosen architecture strikes a balance between capturing the sequential nature of text data and preventing overfitting. Adjustments can be made based on the specific characteristics of the dataset and the task at hand.

## License

- This project is licensed under the MIT License.

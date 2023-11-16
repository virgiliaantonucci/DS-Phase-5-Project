# DS-Phase-5-Project: Emotion Text Processing and Analysis Toolkit
Virgilia Antonucci

## Overview
I am taking on the role of an employee at Twitter. We keep note of the reaction to trending topics. The code determines whether we at Twitter should be running sentiment analysis on the hashtags included in tweets, or if we should ignore them entirely.

I am using a dataset found on Kaggle [here](https://www.kaggle.com/datasets/anjaneyatripathi/emotion-classification-nlp/data). It is comprised of 3099 genuine tweets with an associated emotion (sadness, anger, joy, fear).

This toolkit provides a workflow for loading, processing, and analyzing text data labeled with emotions. It is designed to work with datasets split into training, validation, and test sets. The toolkit includes text cleaning, vectorization, word cloud visualization, common word extraction by emotion, t-SNE visualization, model training, and hyperparameter tuning.

## Getting Started

### Prerequisites
- Python environment (preferably Anaconda)
- Pandas for data manipulation
- NLTK for natural language processing
- Sklearn for modeling and vectorization
- Matplotlib and Seaborn for visualization
- XGBoost for advanced modeling
- Jupyter Notebook or other Python IDE

### Installation
Ensure you have the following libraries installed:

```
pip install pandas matplotlib seaborn nltk scikit-learn wordcloud xgboost
```
```
pip install tensorflow
```
### Usage

#### Preprocess Data
Run the `clean_text` function to preprocess text data. Preprocessing steps include converting text to lowercase, removing URLs, mentions, optional hashtags, tokenizing, removing stop words and punctuation, and stemming.

#### Vectorization
The toolkit supports CountVectorizer for transforming preprocessed text into vector format, necessary for machine learning models.

#### Word Clouds
Generate and visualize word clouds for the overall dataset and for each emotion category, both with and without hashtags.

#### Common Words Analysis
Extract the most common words from the dataset and display them for each emotion.

#### t-SNE Visualization
Visualize high-dimensional vectorized text data in two dimensions using t-SNE (with and without hashtags, of course).

#### Model Training and Evaluation
Train and evaluate Naive Bayes and Support Vector Machine (SVM) classifiers on both cleaned text and texts without hashtags. The toolkit includes examples with accuracy scores and classification reports. The Random Forest model helped the least, so focus should be put on the other models used.

#### Hyperparameter Tuning
Perform hyperparameter tuning for SVM and XGBoost models to find the optimal settings for the TfidfVectorizer and classifier parameters.

## Deep Learning with LSTM for Sentiment Analysis

In addition to traditional machine learning models, this toolkit also utilizes Long Short-Term Memory (LSTM) networks, a type of recurrent neural network, to perform sentiment analysis on tweets. The LSTM models are trained and evaluated separately for datasets with and without hashtags to understand their impact on sentiment analysis.

### LSTM Model for Tweets with Hashtags

- **Tokenization and Sequence Padding:** 
  - The tweets are tokenized using Keras' `Tokenizer` with a vocabulary size of 5000 words and an out-of-vocabulary token `<OOV>`.
  - The tokenized texts are then converted into sequences and padded to a uniform length of 200 tokens for model input.

- **Model Architecture:** 
  - The LSTM model consists of an Embedding layer with an input dimension of 5000 and an output dimension of 64, followed by an LSTM layer with 100 units.
  - The model concludes with a Dense layer for classification into four emotion categories (sadness, anger, joy, fear).

- **Training and Evaluation:** 
  - The model is trained for 5 epochs with a batch size of 64, using early stopping based on validation loss to prevent overfitting.
  - Model performance is evaluated on the test set to determine its effectiveness in sentiment analysis.

### LSTM Model for Tweets without Hashtags

- **Separate Tokenization for Hashtag-less Texts:** 
  - A similar tokenization process is followed for texts without hashtags, ensuring that the model learns from text data unaffected by hashtag context.

- **Model Training and Evaluation:** 
  - The LSTM model architecture remains the same as for the dataset with hashtags.
  - Training and evaluation processes are replicated to assess the impact of excluding hashtags from sentiment analysis.

## Incorporating Pretrained GloVe Embeddings in LSTM Models

### Utilizing GloVe Embeddings

To enhance the sentiment analysis, the toolkit now includes the integration of GloVe (Global Vectors for Word Representation) embeddings, specifically trained on Twitter data. These embeddings provide a more nuanced understanding of language used in tweets.

- **Loading GloVe Embeddings:**
  - The toolkit loads GloVe embeddings (50-dimensional vectors from the `glove.twitter.27B` dataset) to capture the semantic meanings of words more effectively.
  - This is done through a custom function, `load_glove_embeddings`, which reads the GloVe file and creates an embeddings index.

- **Creating an Embedding Matrix:**
  - An embedding matrix is created to map each word in our dataset to its corresponding GloVe vector, if available.
  - This matrix is then used as the weights for the Embedding layer in the LSTM model, providing a rich initialization of word representations.

### LSTM Model Enhancements with GloVe

- **Preprocessing for LSTM:**
  - Text data is tokenized and converted into padded sequences to create a uniform input structure suitable for LSTM processing.

- **LSTM Model Architecture:**
  - The LSTM models (for both datasets with and without hashtags) are now equipped with an Embedding layer initialized with the GloVe embeddings.
  - This layer is set to be non-trainable, ensuring that the pretrained vectors are retained during the training process.

- **Model Training and Evaluation:**
  - Models are trained separately on the datasets with and without hashtags to observe the impact of hashtags on sentiment analysis.
  - Training involves 5 epochs with a batch size of 64, incorporating early stopping based on validation loss.
  - The performance of each model is evaluated on the test set to assess accuracy and loss.

# Conclusion
The model with the highest accuracy was SVM with GridSearch. Between hashtags and no hashtag tweets, tweets with hashtags had a higher accuracy. Therefore, we are Twitter should include hashtags in tweets when we conduct sentiment analysis.

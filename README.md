# DS-Phase-5-Project: Emotion Text Processing and Analysis Toolkit

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

# Conclusion
The model with the highest accuracy was SVM with GridSearch. Between hashtags and no hashtag tweets, tweets with hashtags had a higher accuracy. Therefore, we are Twitter should include hashtags in tweets when we conduct sentiment analysis.

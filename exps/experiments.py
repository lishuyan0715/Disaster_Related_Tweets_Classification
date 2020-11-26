import logging

import pandas as pd

# XGBoost
from xgboost import XGBClassifier

# sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import model_selection
from sklearn.model_selection import train_test_split

from statistics import mean

from data_process import *
# Suppress warnings
import warnings

warnings.filterwarnings('ignore')

# Suppress warnings
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


# NOTE this import needs to happen after the logger is configured

# bag-of-words
def bag_of_words(text):
    """ Convert input text to vectors using BOW model"""
    count_vectorizer = CountVectorizer()
    train_vectors = count_vectorizer.fit_transform(text)
    return train_vectors


# TF-IDF
def tfidf(text):
    """ Convert input text to vectors using TF-IDF model"""
    tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))
    train_tfidf = tfidf.fit_transform(text)
    return train_tfidf


# Fitting a simple Logistic Regression model
def log_reg(vectors, target):
    """ Fit logistic regression model and save the AUC score"""
    clf = LogisticRegression(C=1.0)
    scores = model_selection.cross_val_score(clf, vectors, target, cv=5, scoring="roc_auc")
    return mean(scores)


# Fitting a simple Naive Bayes model
def nb(vectors, target):
    """ Fit naive bayes model and save the AUC score"""
    clf_NB = MultinomialNB()
    scores = model_selection.cross_val_score(clf_NB, vectors, target, cv=5, scoring="roc_auc")
    return mean(scores)


# Fitting a XGBoost Classification model
def xgb(vectors, target):
    """ Fit XGBoost classification model and save the AUC score"""
    clf_xgb = XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8,
                                subsample=0.8, nthread=10, learning_rate=0.1)
    scores = model_selection.cross_val_score(clf_xgb, vectors, target, cv=5, scoring="roc_auc")
    return mean(scores)


if __name__ == '__main__':
    # Load data
    data = pd.read_csv("data/train.csv")
    X = data.iloc[:, :-1]
    y = data['target']

    # Split into train test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    train = X_train
    train['target'] = y_train
    test = X_test
    test['target'] = y_test

    # Preprocess data
    train['text'] = train['text'].apply(lambda x: clean_text(x))
    test['text'] = test['text'].apply(lambda x: clean_text(x))
    train['text'] = train['text'].apply(lambda x: text_preprocessing(x))
    test['text'] = test['text'].apply(lambda x: text_preprocessing(x))

    # Transforming tokens to a vector
    bow_train_vectors = bag_of_words(train['text'])
    bow_test_vectors = bag_of_words(test['text'])
    tfidf_train_vectors = tfidf(train['text'])
    tfidf_test_vectors = tfidf(test['text'])

    #  Run Text Classification models
    # 1. Fitting a simple Logistic Regression on Counts
    bow_lr_score = log_reg(bow_train_vectors, train["target"])

    # 2. Fitting a simple Logistic Regression on TFIDF
    tfidf_lr_score = log_reg(tfidf_train_vectors , train["target"])

    # 3. Fitting a simple Naive Bayes on Counts
    bow_nb_score = nb(bow_train_vectors, train["target"])

    # 4. Fitting a simple Naive Bayes on TFIDF
    tfidf_nb_score = nb(tfidf_train_vectors, train["target"])

    # 5. Fitting a XGBoost Classification on Counts
    bow_xgb_score = xgb(bow_train_vectors, train["target"])

    # 6. Fitting a XGBoost Classification on TFIDF
    tfidf_xgb_score = xgb(tfidf_train_vectors, train["target"])

    # Writing to file
    with open("exps/experiments_auc.txt", "w") as file:
        # Writing data to a file
        file.write("Experiment 1: Fitting a simple Logistic Regression on Counts \nAUC: " + str(bow_lr_score))
        file.write("\nExperiment 2: Fitting a simple Logistic Regression on TFIDF \nAUC: " + str(tfidf_lr_score))
        file.write("\nExperiment 3: Fitting a simple Naive Bayes on Counts \nAUC: " + str(bow_nb_score))
        file.write("\nExperiment 4: Fitting a simple Naive Bayes on TFIDF \nAUC: " + str(tfidf_nb_score))
        file.write("\nExperiment 5: Fitting a XGBoost Classification on Counts \nAUC: " + str(bow_xgb_score))
        file.write("\nExperiment 6: Fitting a XGBoost Classification on TFIDF \nAUC: " + str(tfidf_xgb_score))
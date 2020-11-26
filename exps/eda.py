import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from data_process import *

import warnings

warnings.simplefilter('ignore')


# plot average word/character length distribution
def plot_distribution(tweet):
    """ Plot average word/character length distribution in each tweet"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    word = tweet[tweet['target'] == 1]['text'].str.split().apply(lambda x: [len(i) for i in x])
    sns.distplot(word.map(lambda x: np.mean(x)), ax=ax1, color='blue')
    ax1.set_title('disaster')
    word = tweet[tweet['target'] == 0]['text'].str.split().apply(lambda x: [len(i) for i in x])
    sns.distplot(word.map(lambda x: np.mean(x)), ax=ax2, color='red')
    ax2.set_title('Not disaster')
    fig.suptitle('Average Word Length in Each Tweet')
    fig.savefig('exps/avg_word_length.png')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    word = tweet[tweet['target'] == 1]['text'].str.len()
    sns.distplot(word.map(lambda x: np.mean(x)), ax=ax1, color='blue')
    ax1.set_title('disaster')
    word = tweet[tweet['target'] == 0]['text'].str.len()
    sns.distplot(word.map(lambda x: np.mean(x)), ax=ax2, color='red')
    ax2.set_title('Not disaster')
    fig.suptitle('Average Character Length in Each Tweet')
    fig.savefig('exps/avg_character_length.png')


# create a corpus as a dataframe
def create_corpus_df(tweet, target):
    """ Create a corpus as a dataframe"""
    corpus = []

    for x in tweet[tweet['target'] == target]['text'].str.split():
        for i in x:
            corpus.append(i)
    return corpus


# plot word clouds for disaster tweets
def word_cloud_1(tweet):
    """ Plot word clouds"""
    tweet['text'] = tweet['text'].apply(lambda x: clean_text(x))
    tweet['text'] = tweet['text'].apply(lambda x: text_preprocessing(x))

    corpus_new0 = create_corpus_df(tweet, 1)

    dic0 = {i: corpus_new0.count(i) for i in set(corpus_new0)}
    dic_ranked0 = sorted(dic0.items(), key=lambda x: x[1], reverse=True)
    keys0 = [i[0] for i in dic_ranked0]

    # Generating the wordcloud with the values under the category dataframe
    plt.figure(figsize=(12, 8))
    word_cloud = WordCloud(
        background_color='black',
        max_font_size=80
    ).generate(" ".join(keys0[:50]))
    plt.imshow(word_cloud)
    plt.savefig('exps/disaster_wordcloud.png')


# plot word clouds for non-disaster tweets
def word_cloud_0(tweet):
    """ Plot word clouds"""
    tweet['text'] = tweet['text'].apply(lambda x: clean_text(x))
    tweet['text'] = tweet['text'].apply(lambda x: text_preprocessing(x))

    corpus_new0 = create_corpus_df(tweet, 0)

    dic0 = {i: corpus_new0.count(i) for i in set(corpus_new0)}
    dic_ranked0 = sorted(dic0.items(), key=lambda x: x[1], reverse=True)
    keys0 = [i[0] for i in dic_ranked0]

    # Generating the wordcloud with the values under the category dataframe
    plt.figure(figsize=(12, 8))
    word_cloud = WordCloud(
        background_color='black',
        max_font_size=80
    ).generate(" ".join(keys0[:50]))
    plt.imshow(word_cloud)
    plt.savefig('exps/non_disaster_wordcloud.png')


if __name__ == '__main__':
    # Load data
    tweet = pd.read_csv("data/train.csv")

    # Save distribution and word cloud plots
    plot_distribution(tweet)
    word_cloud_1(tweet)
    word_cloud_0(tweet)

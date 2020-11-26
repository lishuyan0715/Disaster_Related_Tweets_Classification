import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import tensorflow_hub as hub
from bert import tokenization
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score

import bert.tokenization

import warnings

warnings.simplefilter('ignore')

BERT_MODEL_LOCATION = os.environ.get('MODEL_FILE', "models/model.h5")

# encode texts
def bert_encode(texts, tokenizer, max_len=512):
    """ Encode input texts to tokens, masks, and segments that can use to feed into BERT"""
    all_tokens = []
    all_masks = []
    all_segments = []

    for text in texts:
        text = tokenizer.tokenize(text)

        text = text[:max_len - 2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len

        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


# build the model
def build_model(bert_layer, max_len=512):
    """ BUild the model"""
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(clf_output)

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    return model


# load existed model or fit a new model
def load_model(model_existed, train_input=None, train_labels=None):
    """ Load saved model file or fit a new model"""

    module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"
    bert_layer = hub.KerasLayer(module_url, trainable=True)
    model = build_model(bert_layer, max_len=160)

    if model_existed == 1:
        model.load_weights(BERT_MODEL_LOCATION)
    else:
        checkpoint = ModelCheckpoint(BERT_MODEL_LOCATION, monitor='val_loss', save_best_only=True)

        train_history = model.fit(
            train_input, train_labels,
            validation_split=0.2,
            epochs=3,
            callbacks=[checkpoint],
            batch_size=16
        )

    return bert_layer, model


# predict whether the input text is a disaster-related tweet
def disaster(text, bert_layer, model):
    """ Predict whether the input text is a disaster-related tweet, 1 means it is, 0 means it is a non disaster-related
    tweet"""
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

    input = bert_encode([text], tokenizer, max_len=160)
    label = model.predict(input).round().astype(int)

    return str(label[0][0])


# Showing Confusion Matrix
def plot_cm(y_true, y_pred, title, figsize=(5, 5)):
    """ Plot confusion matrix for preducted values and true values, and save the plot"""
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    sns.heatmap(cm, cmap="YlGnBu", annot=annot, fmt='', ax=ax)

    fig.savefig('bert/cm.png')


if __name__ == '__main__':
    # Load layer
    module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"
    bert_layer = hub.KerasLayer(module_url, trainable=True)

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

    # data process
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

    # encode inputs
    train_input = bert_encode(train.text.values, tokenizer, max_len=160)
    test_input = bert_encode(test.text.values, tokenizer, max_len=160)
    train_labels = train.target.values
    test_labels = test.target.values

    # build model
    bert_layer, model = load_model(model_existed=1)

    # predict for the test set and plot confusion matrix
    test_pred = model.predict(test_input).round().astype(int)
    plot_cm(test_pred, y_test, 'Confusion matrix for BERT model', figsize=(7, 7))
    bert_auc = roc_auc_score(y_test, test_pred)

    # Writing to file
    with open("bert/bert_auc.txt", "w") as file:
        # Writing data to a file
        file.write("BERT AUC: " + str(bert_auc))

import os

import pytest
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


@pytest.fixture
def testpath():
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def df_review():
    path_to_csv = os.path.join(testpath(), "data/reviews.csv")
    return pd.read_csv(path_to_csv)


@pytest.fixture
def X_reviews():
    reviews = df_review()['review'].values
    word_tokenizer = Tokenizer(num_words=20000)
    word_tokenizer.fit_on_texts(reviews)

    X = np.zeros((len(reviews), 10, 100), dtype='int32')

    for i, review in enumerate(reviews):
        sentences = sent_tokenize(review)
        tokenized_sentences = word_tokenizer.texts_to_sequences(
            sentences
        )
        tokenized_sentences = pad_sequences(
            tokenized_sentences, maxlen=100
        )

        pad_size = 10 - tokenized_sentences.shape[0]

        if pad_size < 0:
            tokenized_sentences = tokenized_sentences[0:10]
        else:
            tokenized_sentences = np.pad(
                tokenized_sentences, ((0, pad_size), (0, 0)),
                mode='constant', constant_values=0
            )

        # Store this observation as the i-th observation in
        # the data matrix
        X[i] = tokenized_sentences[None, ...]

    return X

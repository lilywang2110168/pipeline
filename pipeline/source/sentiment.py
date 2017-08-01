from __future__ import division

from nltk.corpus import sentiwordnet as swn
from numpy import (average, mean, sign)


def feature_sentiment(descriptors):
    raw_score = mean(map(descriptor_sentiment, descriptors)) if descriptors else 0
    return scale(raw_score, root=3)


def descriptor_sentiment(descriptor):
    if descriptor['token'].pos_ == 'VERB':
        pos = 'v'
    elif descriptor['token'].pos_ == 'ADJ':
        pos = 'a'
    elif descriptor['token'].pos_ == 'NOUN':
        pos = 'n'
    elif descriptor['token'].pos_ == 'ADV':
        pos = 'a'
    else:
        return 0
    synsets = swn.senti_synsets(str(descriptor['token']), pos)
    if not synsets:
        return 0
    sentiment = average(map(synset_sentiment, synsets), weights=[.5 ** e for e in range(len(synsets))])
    if descriptor['negs']:
        sentiment *= -1
    return sentiment


def synset_sentiment(synset):
    return synset.pos_score() - synset.neg_score()


def scale(sentiment, root):
    return 3 + 2 * sign(sentiment) * (abs(sentiment) ** (1 / root))

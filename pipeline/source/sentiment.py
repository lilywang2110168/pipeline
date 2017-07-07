from nltk.corpus import sentiwordnet as swn
from numpy import (mean, sign, sqrt)


def feature_sentiment(descriptors):
    return scale(mean(map(descriptor_sentiment, descriptors)))


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
    sentiment = mean(map(synset_sentiment, synsets))
    if descriptor['negs']:
        sentiment *= -1
    return sentiment


def synset_sentiment(synset):
    return synset.pos_score() - synset.neg_score()


def scale(sentiment):
    return 3 + 2 * sign(sentiment) * sqrt(sqrt(abs(sentiment)))

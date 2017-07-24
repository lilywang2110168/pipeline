from argparse import ArgumentParser
import codecs
import cPickle as pickle
from functools import partial
import itertools
import os

import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from scipy import sparse
from sklearn import naive_bayes
from sklearn.externals import joblib
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score, f1_score)
from sklearn.model_selection import StratifiedKFold

OBJECTIVE = 0
SUBJECTIVE = 1


def preprocess_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    for i, sent in enumerate(tokens):
        for j, token in enumerate(sent):
            token = lemmatizer.lemmatize(token, pos='v')
            token = token.encode('ascii', 'ignore')
            token = token.lower()
            if token.isdigit():
                token = '0'
            tokens[i][j] = token
    return tokens


def feature_matrix(tokens, dictionary):
    X = sparse.dok_matrix((len(tokens), len(dictionary)), dtype=np.int8)
    default_value = dictionary['*UNK*']
    for sent_index, sent in enumerate(tokens):
        for word in sent:
            word_index = dictionary.get(word, default_value)
            X[sent_index, word_index] = 1
    return X.tocsr()  # more efficient to slice and multiply


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-id', dest='model_id', default='default')
    args = parser.parse_args()

    path = os.path.join(os.path.dirname(__file__), '..', 'resources', 'plot_sentences.txt')
    with open(path) as f:
        objective_sentences = list(f)

    subjective_sentences = []
    opinion_folder = os.path.join(os.path.dirname(__file__), '..', 'resources', 'opinion_sentences')
    for file_name in os.listdir(opinion_folder):
        with codecs.open(os.path.join(opinion_folder, file_name), 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                subjective_sentences.append(line)

    # tokenizing
    objective_tokens = map(nltk.word_tokenize, objective_sentences)
    subjective_tokens = map(nltk.word_tokenize, subjective_sentences)
    tokens = objective_tokens + subjective_tokens

    # preprocessing
    tokens = preprocess_tokens(tokens)

    # constructing the dictionary
    dictionary = {}
    c = itertools.count()
    for sent in tokens:
        for word in sent:
            if word not in dictionary:
                dictionary[word] = next(c)
    dictionary['*UNK*'] = next(c) # unknown values

    # constructing the matrices
    X = feature_matrix(tokens, dictionary)
    Y = np.zeros(len(tokens), dtype=np.int8)
    Y[:len(objective_sentences)] = OBJECTIVE
    Y[len(objective_sentences):] = SUBJECTIVE

    # Naive Bayes classifier
    clf = naive_bayes.BernoulliNB()

    skf = StratifiedKFold(n_splits=5)
    confusion_matrices = []
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    for train, test in skf.split(X, Y):
        xtrain, xtest = X[train], X[test]
        ytrain, ytest = Y[train], Y[test]
        clf.fit(xtrain, ytrain)
        ypredict = clf.predict(xtest)
        confusion_matrices.append(confusion_matrix(ytest, ypredict))
        accuracies.append(accuracy_score(ytest, ypredict))
        precisions.append(precision_score(ytest, ypredict))
        recalls.append(recall_score(ytest, ypredict))
        f1s.append(f1_score(ytest, ypredict))

    print '5-fold cross-validation'
    print 'sum of confusion matrices'
    print sum(confusion_matrices)
    print 'average accuracy'
    print np.mean(accuracies)
    print 'average precision'
    print np.mean(precisions)
    print 'average recall'
    print np.mean(recalls)
    print 'average f1'
    print np.mean(f1s)

    # train a classifier on the full training set
    final_classifier = naive_bayes.BernoulliNB()
    final_classifier.fit(X, Y)

    save_folder = os.path.join(os.path.dirname(__file__), '..', 'models', args.model_id)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    with open(os.path.join(save_folder, 'classifier'), 'w+') as f:
        joblib.dump(final_classifier, f)
    with open(os.path.join(save_folder, 'dictionary'), 'w+') as f:
        pickle.dump(dictionary, f)
    print 'Saved trained classifier at {}'.format(os.path.abspath(save_folder))

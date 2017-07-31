from collections import defaultdict
from functools import partial
from itertools import permutations

from nltk.stem import PorterStemmer


def get_final_feature_descriptors(nltk_features, dep_features):
    stemmer = PorterStemmer()
    nltk_features_stemmed = [feat.split(' ') for feat in nltk_features]
    nltk_features_stemmed = map(partial(map, stemmer.stem), nltk_features_stemmed)
    for dep_feat in dep_features:
        dep_feat['token_stemmed'] = stemmer.stem(str(dep_feat['token']))
        dep_feat['compounds_stemmed'] = map(str, dep_feat['compounds'])
        dep_feat['compounds_stemmed'] = map(stemmer.stem, dep_feat['compounds_stemmed'])

    filtered = defaultdict(list)
    for nltk_feat, nltk_feat_stemmed in zip(nltk_features, nltk_features_stemmed):
        for dep_feat in dep_features:
            if nltk_feat_equals_dep_feat(nltk_feat_stemmed, dep_feat):
                filtered[nltk_feat] += dep_feat['descriptors']
    return filtered


def nltk_feat_equals_dep_feat(nltk_feat_stemmed, dep_feat):
    num_compounds = len(nltk_feat_stemmed) - 1
    for compounds_stemmed in permutations(dep_feat['compounds_stemmed'], num_compounds):
        compound_dep_feat = list(compounds_stemmed) + [dep_feat['token_stemmed']]
        if compound_dep_feat == nltk_feat_stemmed:
            return True
    return False


def dependency_features(doc):
    '''
    output ::= list<occurance>
    occurance ::= {'token': token, 'compunds': list<token>, 'descriptors': list<descriptor>}
    descriptor ::= {'token': token, 'negs': list<token>, 'advs': list<token>}
    '''

    descriptor_map = defaultdict(list)  # feat_index -> list<desc_index>
    compound_map = defaultdict(list)  # feat_index -> list<compound_index>
    neg_map = defaultdict(list)  # desc_index -> list<neg_index>
    adv_map = defaultdict(list)  # desc_index -> list<adv_index>

    for token in doc:
        if token.dep_ == 'amod':
            descriptor_map[token.head.i].append(token.i)
        elif token.dep_ in ['dobj', 'nmod', 'nsubj']:
            descriptor_map[token.i].append(token.head.i)
        elif token.dep_ == 'compound':
            compound_map[token.head.i].append(token.i)
        elif token.dep_ == 'neg':
            neg_map[token.head.i].append(token.i)
        elif token.dep_ == 'advmod':
            adv_map[token.head.i].append(token.i)

    occurances = []
    for feat_index, desc_indices in descriptor_map.iteritems():
        descriptors = []
        feat_negs = map(doc.__getitem__, neg_map[feat_index])
        for desc_index in desc_indices:
            negs = map(doc.__getitem__, neg_map[desc_index])
            advs = map(doc.__getitem__, adv_map[desc_index])
            descriptors.append({'token': doc[desc_index], 'negs': negs + feat_negs, 'advs': advs})
        compounds = map(doc.__getitem__, compound_map[feat_index])
        occurance = {'token': doc[feat_index], 'compounds': compounds, 'descriptors': descriptors}
        occurances.append(occurance)

    return occurances

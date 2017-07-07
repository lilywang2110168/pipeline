from collections import defaultdict
from itertools import permutations

import nltk


def get_final_feature_descriptors(nltk_features, dep_features):
    filtered = defaultdict(list)
    for dep_feat in dep_features:
        indices = feat_in(dep_feat, nltk_features)
        for index in indices:
            filtered[nltk_features[index]] += dep_feat['descriptors']
    return filtered


def feat_in(dep_feat, feats):
    indices = []
    for index, feat in enumerate(feats):
        if dep_feat_equals(dep_feat, feat):
            indices.append(index)
    return indices


def dep_feat_equals(dep_feat, feat):
    if not dep_feat['compounds']:
        return feat_equals(dep_feat['word'], feat)
    for num_compounds in range(1, len(dep_feat['compounds']) + 1):
        for compounds in permutations(dep_feat['compounds'], num_compounds):
            compound_feat = ' '.join(compounds) + ' ' + dep_feat['word']
            if feat_equals(compound_feat, feat):
                return True
    return False


def feat_equals(feat1, feat2):
    split1 = feat1.split(' ')
    split2 = feat2.split(' ')
    if len(split1) != len(split2):
        return False
    stemmer = nltk.stem.PorterStemmer()
    stem1 = [stemmer.stem(w) for w in split1]
    stem2 = [stemmer.stem(w) for w in split2]
    equal = map(lambda x: x[0] == x[1], zip(stem1, stem2))
    return reduce(lambda x, y: x and y, equal)


def dependency_features(nlp, text):
    '''
    output ::= list<occurance>
    occurance ::= {'token': token, 'compunds': list<token>, 'descriptors': list<descriptor>}
    descriptor ::= {'token': token, 'negs': list<token>, 'advs': list<token>}
    '''

    doc = nlp(unicode(text))

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
        feat_negs = map(lambda i: doc[i], neg_map[feat_index])
        for desc_index in desc_indices:
            negs = map(lambda i: doc[i], neg_map[desc_index])
            advs = map(lambda i: doc[i], adv_map[desc_index])
            descriptors.append({'token': doc[desc_index], 'negs': negs + feat_negs, 'advs': advs})
        compounds = map(lambda i: doc[i], compound_map[feat_index])
        occurance = {'token': doc[feat_index], 'compounds': compounds, 'descriptors': descriptors}
        occurances.append(occurance)

    return occurances

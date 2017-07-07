import os

import spacy

from pipeline.source import depparse
from pipeline.source import extract

with open('../resources/laptop_features.txt') as f:
    nltk_feats = [line[:line.index(':')] for line in f]

nlp = spacy.load('en')

path = os.path.join(os.path.dirname(__file__), '../resources/laptop_reviews.txt')
reviews = extract.reviews_from_file(path, splitter='\n\n')

dep_feats = depparse.dependency_features(nlp, '\n\n'.join(reviews[:1000]))

result = depparse.get_final_feature_descriptors(nltk_feats, dep_feats)

for k, v in result.iteritems():
    for desc in v:
        if desc['negs']:
            print '{}: {}'.format(k, desc)

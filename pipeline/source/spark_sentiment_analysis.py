from operator import itemgetter
import os

import spacy
import depparse, sentiment

path = os.path.join(os.path.dirname(__file__), '../resources/laptop_reviews.txt')
reviews = extract.reviews_from_file(path, splitter='\n\n')
with open('../resources/laptop_features.txt') as f:
    nltk_feats = [line[:line.index(':')] for line in f]

nlp = spacy.load('en')
doc = nlp(unicode('\n\n'.join(reviews[:1000])))
dep_feats = depparse.dependency_features(doc)
result = depparse.get_final_feature_descriptors(nltk_feats, dep_feats)
sentiments = [(feat, sentiment.feature_sentiment(descs)) for feat, descs in result.iteritems()]
for feat, sentiment in sorted(sentiments, key=itemgetter(1), reverse=True):
    print feat, sentiment

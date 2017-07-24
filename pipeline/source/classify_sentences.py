# Usage:
# python classify_sentences.py path/to/input_file_name output_file_name [--model-id ID]

from argparse import ArgumentParser
import cPickle as pickle
from functools import partial
import json
import os

import nltk
from sklearn.externals import joblib

from train_sentence_classifier import (preprocess_tokens, feature_matrix, SUBJECTIVE)

parser = ArgumentParser()
parser.add_argument('input_file')
parser.add_argument('output_file')
parser.add_argument('--model-id', dest='model_id', default='default', metavar='ID')
args = parser.parse_args()


def is_json(myjson):
    try:
        json.loads(myjson)
    except ValueError:
        return False
    return True


model_folder = os.path.join(os.path.dirname(__file__), '..', 'models', args.model_id)
with open(os.path.join(model_folder, 'classifier')) as f:
    clf = joblib.load(f)
with open(os.path.join(model_folder, 'dictionary')) as f:
    dictionary = pickle.load(f)

output_folder = os.path.join(os.path.dirname(__file__), '..', 'output')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

input_path = os.path.abspath(args.input_file)
output_path = os.path.join(output_folder, args.output_file)

sents = 0
subj_sents = 0

with open(input_path) as input_file, open(output_path, 'w+') as output_file:
    print 'File has {} lines'.format(sum(1 for _ in input_file))
    input_file.seek(0)
    for i, line in enumerate(input_file):
        if i % 1000 == 0:
            print 'line {}'.format(i)
        if is_json(line):
            obj = json.loads(line)
            sentences = nltk.sent_tokenize(obj['reviewText'])
            tokens = map(nltk.word_tokenize, sentences)
            tokens = preprocess_tokens(tokens)
            if len(tokens) > 0:
                X = feature_matrix(tokens, dictionary)
                ypredict = clf.predict(X)
                subjective_sentences = [sent for sent, pred in zip(sentences, ypredict) if pred == SUBJECTIVE]
                obj['reviewText'] = ' '.join(subjective_sentences)
                json.dump(obj, output_file)
            sents += len(sentences)
            subj_sents += len(subjective_sentences)

print '{} sentences were classified as subjective out of {} total sentences'.format(subj_sents, sents)
print 'Wrote output to {}'.format(os.path.abspath(output_path))

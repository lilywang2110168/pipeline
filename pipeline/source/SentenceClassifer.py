import nltk
from nltk.stem import WordNetLemmatizer
import codecs
from sklearn import svm
from sklearn import naive_bayes
from sklearn.model_selection import (train_test_split, StratifiedKFold)
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score, f1_score)
from sklearn.externals import joblib
from scipy import sparse
import numpy as np
import itertools
import os
import glob
from argparse import ArgumentParser
import cPickle
import json

parser = ArgumentParser()
parser.add_argument('--save-folder', dest='save_folder')
args = parser.parse_args()

wordnet_lemmatizer = WordNetLemmatizer()

sentencesPlot = []
sentencesQuote = []

# importing files
with open("plot.tok.gt9.5000") as f:
    for line in f:
        sentencesPlot.append(line)
'''
with codecs.open("quote.tok.gt9.5000", "r", encoding="utf-8", errors='ignore') as f2:
    for line in f2:
        sentencesQuote.append(line)


'''
sentencesQuote = []


os.chdir("topics")
for file in glob.glob("*.data"):
    with codecs.open(file, "r", encoding="utf-8", errors='ignore') as f2:
        for line in f2:
            sentencesQuote.append(line)

os.chdir("..")

#print sentencesQuote
#print len(sentencesQuote)
'''

os.chdir("..")
with codecs.open('Wikipedia.txt', "r",encoding="utf-8", errors='ignore') as f1:
    data=f1.read().replace('\n', '')

sentencesPlot = nltk.sent_tokenize(data)

print len(sentencesPlot)
'''


# tokenizing
tokensPlot = [nltk.word_tokenize(sent) for sent in sentencesPlot]
#print tokensPlot
tokensQuote = [nltk.word_tokenize(sent) for sent in sentencesQuote]
#print tokensQuote
tokens = tokensPlot + tokensQuote



#preprocessing
for i in range(len(tokens)): 
    for j in range(len(tokens[i])):
        tmp=wordnet_lemmatizer.lemmatize(tokens[i][j], pos='v')
        tmp=tmp.encode('ascii', 'ignore')
        tmp=tmp.lower()
        
        if tmp.isdigit():
            tmp='0'

        tokens[i][j]=tmp


# constructing the dictionary
dictionary = {}
c = itertools.count()
for sent in tokens:
    for word in sent:
        if word not in dictionary:
            dictionary[word] = next(c)

#print dictionary

# constructing the matrices
X = sparse.dok_matrix( (len(tokens),len(dictionary)) , dtype=np.int8 )
Y = np.zeros(len(tokens), dtype=np.int8)

for i, sent in enumerate(tokensPlot):
    for word in sent:
        index = dictionary[word]
        X[i,index] = 1
    Y[i] = 1

for i, sent in enumerate(tokensQuote, start=len(tokensPlot)):
    for word in sent:
        index = dictionary[word]
        X[i,index] = 1


# xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.20, random_state=42)

clf = naive_bayes.BernoulliNB()

# clf.fit(xtrain, ytrain)
# print 'train accuracy'
# print clf.score(xtrain, ytrain)
# print 'test accuracy'
# print clf.score(xtest, ytest)
#
# ypredict = clf.predict(xtest)
# print 'test confusion matrix'
# print confusion_matrix(ytest, ypredict)

skf = StratifiedKFold(n_splits=5)
confusion_matrices = []
accuracies = []
precisions = []
recalls = []
f1s = []
X = X.tocsr() # convert sparse matrix to a more efficient structure for slicing
for train,test in skf.split(X,Y):
    xtrain,xtest = X[train],X[test]
    ytrain,ytest = Y[train],Y[test]
    clf.fit(xtrain,ytrain)
    ypredict = clf.predict(xtest)
    confusion_matrices.append( confusion_matrix(ytest, ypredict) )
    accuracies.append( accuracy_score(ytest, ypredict) )
    precisions.append( precision_score(ytest,ypredict) )
    recalls.append( recall_score(ytest, ypredict) )
    f1s.append( f1_score(ytest, ypredict))



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

if args.save_folder:
    save_folder = os.path.abspath(args.save_folder)
    final_classifier = naive_bayes.BernoulliNB()
    final_classifier.fit(X,Y)
    with open(save_folder + '/classifier', 'w+') as f:
        joblib.dump(final_classifier, f)
    with open(save_folder + '/dictionary', 'w+') as f:
        cPickle.dump(dictionary, f)
    print 'Saved trained classifier at {}'.format(os.path.abspath(args.save_folder))


sentences=[]



#Using the trained model to classify reviews (by sentence)

def is_json(myjson):
    try:
        json_object = json.loads(myjson)
    except ValueError, e:
        return False
    return True


#importing JSON File

with open('reviews_Laptops_test.json') as f1:
    outputFile_subjective = open('reviews_Laptops_test_subjSen_Ruchi.json', 'w')
    outputFile_objective = open('reviews_Laptops_test_objSen_Ruchi.json', 'w')
    #fileObj = open('ObjReviews2_ruchi.json', 'w')
    #fileSub = open('SubReviews2_ruchi.json', 'w')
    for line in f1:
        if is_json(line):
            jline = json.loads(line)
            review = jline['reviewText']
            #separates review by sentences
            reviewSent = nltk.sent_tokenize(review)
            tokensReviewList = []
            for sentence in reviewSent:
                tokensReview = nltk.word_tokenize(sentence)
                #for each word in sentence lemmatizes that word
                for i in range(len(tokensReview)):
                    tmp =  wordnet_lemmatizer.lemmatize(tokensReview[i], pos='v')
                    tmp = tmp.encode('ascii', 'ignore')
                    tmp = tmp.lower()
                    tokensReview[i] = tmp
                tokensReviewList.append(tokensReview)
            if len(tokensReviewList) > 0:
                X = sparse.dok_matrix((len(tokensReviewList), len(dictionary)), dtype=np.int8)
                for i,sent in enumerate(tokensReviewList):
                    for word in sent:
                        if word in dictionary:
                            index = dictionary[word]
                            X[i,index] = 1
                ypredict = clf.predict(X)
                subjectivePar = ""
                objectivePar = ""
                for i in range(len(ypredict)):
                    if ypredict[i] == 0:
                        objectivePar = objectivePar + reviewSent[i]
                    else:
                        subjectivePar = subjectivePar + reviewSent[i]
                #first writeout subjective input 
                print "subjectivePar", subjectivePar
                jline['reviewText'] = subjectivePar
                outputLine = json.dumps(jline)
                outputFile_subjective.write(outputLine + "\n")
                #next write out objective input 
                print "objectivePar", objectivePar
                jline['reviewText'] = objectivePar
                outputLine = json.dumps(jline)
                outputFile_objective.write(outputLine + "\n")

print "Done!"





































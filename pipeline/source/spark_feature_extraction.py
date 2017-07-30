import nltk
import pyspark

from feature_extraction import ( getBigrams, pruneFeature, getRepresentativeFeatures,
                                                getTopFeatures)
from spark import (get_sc, load_table)



sc = get_sc()
spark = pyspark.sql.SparkSession(sc)

#globals
ps = nltk.stem.PorterStemmer()
lemmatizer = nltk.stem.WordNetLemmatizer()
≈


load_table(spark, 'AmazonReviews')
df = spark.sql('SELECT reviewText from AmazonReviews')

#parallel programming don't need to prallelize an existing dataframe
##reviews=sc.parallelize(df)

grammar = r"""
 NP: {<NN><NN>}   # nouns and nouns
    {<JJ><NN>}          # ajetives and nouns
"""

cp = nltk.RegexpParser(grammar)

sentences = [ str(i.reviewText) for i in df.collect()]
reviews=sc.parallelize(sentences)

#not doing sentence tokenizer
tokens=reviews.map(lambda x:nltk.word_tokenize(x).map(lambda x:nltk.pos_tag(x)).take(500)
result=tokens.map(lambda x:cp.parse(x))

print tokens

##print tokens.map(lambda x: getUnigrams(x)).collect()


                         
'''



tokens=df.rdd.map(lambda x:nltk.word_tokenize(str(x.reviewText))).map(lambda x:nltk.pos_tag(x)).collect()
result=tokens.map(lambda x:cp.parse(x))

dictionary = getUnigrams(tokens)
dictionaryPhrases = getBigrams(result)
deleteSingle, deletePhrase = pruneFeature(dictionary, dictionaryPhrases)

print "there"

for item in deleteSingle:
    if item in dictionary:
        del dictionary[item]
for item in deletePhrase:
    if item in dictionaryPhrases:
        del dictionaryPhrases[item]
dictionary = getRepresentativeFeatures(dictionary, 10)
dictionaryPhrases = getRepresentativeFeatures(dictionaryPhrases, 5)
myList = getTopFeatures(dictionary, 10)
print myList
myList2 = getTopFeatures(dictionaryPhrases, 20)
print myList2



reviews = [ str(i.reviewText) for i in df.collect()]
sentences=[]
for line in reviews:
  sents = nltk.sent_tokenize(line)
  for sent in sents:
    sentences.append(sent)
  
reviews=sc.parallelize(sentences)
mydata.map(lambda x: x.split('\t')).\
    map(lambda y: (y[0], y[2], y[1]))

reviews = [ str(i.reviewText) for i in df.collect()]

sentences=[]
for line in reviews:
  sents = nltk.sent_tokenize(line)
  for sent in sents:
    sentences.append(sent)



print "I am here"

tokens = [nltk.word_tokenize(sent) for sent in sentences]
tokens = [nltk.pos_tag(sent) for sent in tokens]

print "now I am here"
grammar = r"""
 NP: {<NN><NN>}   # nouns and nouns
    {<JJ><NN>}          # ajetives and nouns
"""
cp = nltk.RegexpParser(grammar)
result = [cp.parse(sent) for sent in tokens]
dictionary = getUnigrams(tokens)
dictionaryPhrases = getBigrams(result)
deleteSingle, deletePhrase = pruneFeature(dictionary, dictionaryPhrases)

print "there"

for item in deleteSingle:
    if item in dictionary:
        del dictionary[item]
for item in deletePhrase:
    if item in dictionaryPhrases:
        del dictionaryPhrases[item]
dictionary = getRepresentativeFeatures(dictionary, 10)
dictionaryPhrases = getRepresentativeFeatures(dictionaryPhrases, 5)
myList = getTopFeatures(dictionary, 10)
print myList
myList2 = getTopFeatures(dictionaryPhrases, 20)
print myList2

'''

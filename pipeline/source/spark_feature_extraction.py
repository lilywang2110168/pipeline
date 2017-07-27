import nltk
import pyspark

from feature_extraction import (getUnigrams, getBigrams, pruneFeature, getRepresentativeFeatures,
                                                getTopFeatures)
from spark import (get_sc, load_table)

sc = get_sc()
spark = pyspark.sql.SparkSession(sc)
load_table(spark, 'AmazonReviews')
df = spark.sql('SELECT reviewText from AmazonReviews')
df.show()


reviews = [ df.reviewText for i in df.collect()]

 print reviews
'''
tokens = map(nltk.word_tokenize, reviews)
grammar = r"""
 NP: {<NN><NN>}   # nouns and nouns
    {<JJ><NN>}          # ajetives and nouns
"""
cp = nltk.RegexpParser(grammar)
result = [cp.parse(sent) for sent in tokens]
dictionary = getUnigrams(tokens)
dictionaryPhrases = getBigrams(result)
deleteSingle, deletePhrase = pruneFeature(dictionary, dictionaryPhrases)
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

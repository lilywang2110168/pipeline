import nltk
import pyspark
from multiprocessing import Pool
import time

from feature_extraction import ( getUnigrams, getBigrams, pruneFeature, getRepresentativeFeatures,
                                                getTopFeatures)
from spark import (get_sc, load_table)


start_time = time.time()
sc = get_sc()
spark = pyspark.sql.SparkSession(sc)

#globals
ps = nltk.stem.PorterStemmer()
lemmatizer = nltk.stem.WordNetLemmatizer()


load_table(spark, 'LilyLaptopReviews')
df = spark.sql('SELECT reviewText from LilyLaptopReviews')


def parseGrammar(sent):
  grammar = r"""
    NP: {<NN><NN>}   # nouns and nouns
    {<JJ><NN>}          # ajetives and nouns
      """
  cp = nltk.RegexpParser(grammar)
  return cp.parse(sent)
  
def getSentences(row):
  return str(row.reviewText)
    
pool = Pool(16) 

sentences=pool.map(getSentences, df.collect())
print("--- %s seconds ---loading data" % (time.time() - start_time))
start_time = time.time()

tokens = pool.map(nltk.word_tokenize, sentences)
tokens= pool.map(nltk.pos_tag, tokens)
print("-- %s seconds ---tokenizing and POS tagging" % (time.time() - start_time))
start_time = time.time()
result=pool.map(parseGrammar, tokens)
pool.close() 
pool.join()

print("--- %s seconds ---grammer" % (time.time() - start_time))
start_time = time.time()
pool.close() 
pool.join() 

dictionary=getUnigrams(tokens)
dictionaryPhrases = getBigrams(result)
print("--- %s seconds ---get uimgrams and bigrams" % (time.time() - start_time))
start_time = time.time()
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
myList2 = getTopFeatures(dictionaryPhrases, 20)
print myList2+myList

print("--- %s seconds ---the rest" % (time.time() - start_time))
                 
'''

##USING SPARK MAP...
sentences = [ str(i.reviewText) for i in df.collect()]
reviews=sc.parallelize(sentences)


#not doing sentence tokenizer
tokens=reviews.map(lambda x:nltk.word_tokenize(x)).map(lambda x:nltk.pos_tag(x)).collect()
result=sc.parallelize(tokens).map(lambda x:cp.parse(x)).collect()


##USING RDD MAP WHICH DOES NOT UTINIZE MULTITHREADING
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


tokens = [nltk.word_tokenize(sent) for sent in sentences]
tokens = [nltk.pos_tag(sent) for sent in tokens]
'''

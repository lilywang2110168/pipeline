from operator import itemgetter
from spark import (get_sc, load_table)
import pyspark
import os
from multiprocessing import Pool
import time


import spacy
import depparse, sentiment,extract



sc = get_sc()
spark = pyspark.sql.SparkSession(sc)
load_table(spark, 'AmazonReviews')
df = spark.sql('SELECT reviewText from AmazonReviews')

def getSentences(row):
  return str(row.reviewText)
    
pool = Pool(16) 

reviews=pool.map(getSentences, df.collect())
with open('../resources/laptop_features.txt') as f:
    nltk_feats = [line[:line.index(':')] for line in f]

print nltk_feats    

start_time = time.time()
nlp = spacy.load('en')
doc = nlp(unicode('\n\n'.join(reviews[:5000])))

print("--- %s seconds ---joining reviews" % (time.time() - start_time))
start_time = time.time()


dep_feats = depparse.dependency_features(doc)


print("--- %s seconds ---getting def_feats" % (time.time() - start_time))
start_time = time.time()

print dep_feats
result = depparse.get_final_feature_descriptors(nltk_feats, dep_feats)



print("--- %s seconds ---getting feature_descriptor" % (time.time() - start_time))
start_time = time.time()
sentiments = [(feat, sentiment.feature_sentiment(descs)) for feat, descs in result.iteritems()]
for feat, sentiment in sorted(sentiments, key=itemgetter(1), reverse=True):
    print feat, sentiment

print("--- %s seconds ---getting sentimient" % (time.time() - start_time))
start_time = time.time()

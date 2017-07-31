from operator import itemgetter
from spark import (get_sc, load_table)
import pyspark
import os
from multiprocessing import Pool
import time
import json
import spacy
import depparse, sentiment,extract



sc = get_sc()
spark = pyspark.sql.SparkSession(sc)

load_table(spark, 'LilyLaptopReviews')
df = spark.sql('SELECT reviewText from LilyLaptopReviews')

def getSentences(row):
  return str(row.reviewText)
    
pool = Pool(16) 

reviews=pool.map(getSentences, df.collect())

##getting the list of features
with open('features.txt') as f:
  for line in f:
    jline=json.loads(line)
    nltk_feats = [str(item['featureName']) for item in jline['features']]

print nltk_feats    

start_time = time.time()
nlp = spacy.load('en')
doc = nlp(unicode('\n\n'.join(reviews[:5000])))

print("--- %s seconds ---joining reviews" % (time.time() - start_time))
start_time = time.time()


dep_feats = depparse.dependency_features(doc)


print("--- %s seconds ---getting def_feats" % (time.time() - start_time))
start_time = time.time()

result = depparse.get_final_feature_descriptors(nltk_feats, dep_feats)



print("--- %s seconds ---getting feature_descriptor" % (time.time() - start_time))
start_time = time.time()
sentiments = [(feat, sentiment.feature_sentiment(descs)) for feat, descs in result.iteritems()]

print sentiments

for feat, sentiment in sorted(sentiments, key=itemgetter(1), reverse=True):
    print feat, sentiment

    
    
   

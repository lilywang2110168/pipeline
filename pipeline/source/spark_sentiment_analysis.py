from operator import itemgetter
from spark import (get_sc, load_tableISCDM)
import pyspark
import os
from multiprocessing import Pool
import time
import json
import spacy
import depparse, sentiment,extract


##globals
jline={}
sc = get_sc()
spark = pyspark.sql.SparkSession(sc)

load_tableISCDM(spark, 'h')
df = spark.sql('SELECT reviewText from h')

def getSentences(row):
  return str(row.reviewText)
    
pool = Pool(16) 

reviews=pool.map(getSentences, df.collect())

##getting the list of features
with open('features_headphones.json') as f:
  for line in f:
    jline=json.loads(line)
    nltk_feats = [str(feat[0]) for feat in jline['features']]

print nltk_feats    

start_time = time.time()
nlp = spacy.load('en')
doc = nlp(unicode('\n\n'.join(reviews[:1000])))

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
myFile=open('sentiment_headphones.json', 'w')
for feat, sentiment in sentiments:
  jline['features'][feat]['sentimentScore']=sentiment

json.dump(jline, myFile)
        
      

    
    
   

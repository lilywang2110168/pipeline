
from spark import (get_sc, load_tableDatabase, load_tableISCDM)
from multiprocessing import Pool
import pyspark
import os
import time
import json
import depparse, sentiment,extract
import spacy

sc = get_sc()
spark = pyspark.sql.SparkSession(sc)

load_tableISCDM(spark, 'LilyLaptopReviews')
load_tableDatabase(spark, 'Category_features')


df = spark.sql("SELECT features_featureName from Category_features where Category='laptops'")
nltk_feats=[str(i.features_featureName) for i in df.collect()]
print nltk_feats
'''

nltk_feats=['size', 'screen resolution', 'number pad', 'desktop replacement', 'hard drive', 'touch screen', 'speed', 'port', 'wireless mouse', 'build quality', 'sound quality', 'desktop', 'machine', 'window', 'program', 'speaker', 'power cord', 'screen size', 'power button', 'backlit keyboard', 'customer service', 'word processing', 'video card', 'graphic', 'operating system', 'button', 'tech support', 'battery life', 'light weight', 'optical drive', 'mouse pad', 'software']

df = spark.sql("SELECT reviewText, ID from LilyLaptopReviews")
df.show()
nlp = spacy.load('en')
myFile=open('reviewAnalysis.txt', 'w')

data={}

def senti_analysis(i):
  doc = nlp(unicode(i.reviewText))
  dep_feats = depparse.dependency_features(doc)
  result = depparse.get_final_feature_descriptors(nltk_feats, dep_feats)
  sentiments = [(feat, sentiment.feature_sentiment(descs)) for feat, descs in result.iteritems()]
  if(len(sentiments)>0):
    data={}
    data['features']=[]
    data["ID"]=i.ID
    for item in sentiments:
      tmp={}
      tmp["featureName"]=item[0]
      tmp["sentimentScore"]=item[1]
      data['features'].append(tmp)
    return data
  
pool = Pool(16) 
data=pool.map(senti_analysis, df.collect())


for item in data:
  if(item!=None):
    json.dump(item,myFile)
    myFile.write('\n')
  
  
    
'''    
  
  
  
  

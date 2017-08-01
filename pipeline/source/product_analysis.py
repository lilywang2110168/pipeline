
from spark import (get_sc, load_tableDatabase, load_tableISCDM)
from multiprocessing import Pool
import pyspark
import os
import time
import json
import depparse, sentiment,extract
import spacy
import numpy


sc = get_sc()
spark = pyspark.sql.SparkSession(sc)

load_tableDatabase(spark, 'ReviewAnalysis_features')
load_tableDatabase(spark, 'ReviewAnalysis')

df=spark.sql('SELECT ReviewAnalysis.product, ReviewAnalysis.reviewId, ReviewAnalysis_features.features_featureName, ReviewAnalysis_features.features_sentimentScore FROM ReviewAnalysis INNER JOIN ReviewAnalysis_features ON ReviewAnalysis_features.ReviewAnalysis=ReviewAnalysis.reviewID')
df.show()

dictionary={}
for i in df.collect():
  if i.product not in dictionary:
    dictionary[i.product]={}
  if i.features_featureName not in  dictionary[i.product]:
    dictionary[i.product][i.features_featureName]=[]
  dictionary[i.product][i.features_featureName].append(float(i.features_sentimentScore))

myFile=open('productAnalysis.txt', 'w')

for key in dictionary:
  data={}
  data['productID']=key
  data['features']=[]
  features=dictionary[key]
  for key in features:
    tmp={}
    tmp['featureName']=key
    tmp["sentimentScore"]=numpy.mean(features[key])
    tmp["popularityScore"]=len(features[key])
    data['features'].append(tmp)
  json.dump(data, myFile)
  print data
  myFile.write('\n')
  
    
  
    
    
    
  
     
      
  
  

##create a dictionary of productID pointing to reviewIDs???


##then how do you get the features associated with the reviews????

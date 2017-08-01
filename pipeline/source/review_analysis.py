
from spark import (get_sc, load_tableDatabase, load_tableISCDM)
import pyspark
import os
import time
import json

sc = get_sc()
spark = pyspark.sql.SparkSession(sc)

load_tableISCDM(spark, 'LilyLaptopReviews')
load_tableDatabase(spark, 'Category')

df = spark.sql("SELECT features, categoryName from Category where categoryName='laptops'")
df.show()
for i in df.collect():
  print i.features
  print type(i.features)

featureList=set(['size', 'screen resolution', 'number pad', 'desktop replacement', 'hard drive', 'touch screen', 'speed', 'port', 'wireless mouse', 'build quality', 'sound quality', 'desktop', 'machine', 'window', 'program', 'speaker', 'power cord', 'screen size', 'power button', 'backlit keyboard', 'customer service', 'word processing', 'video card', 'graphic', 'operating system', 'button', 'tech support', 'battery life', 'light weight', 'optical drive', 'mouse pad', 'software'])

df = spark.sql("SELECT reviewText, ID from LilyLaptopReviews")
df.show()

for i in df.collect():
  review=i.reviewText
  doc = nlp(unicode(review))
  
  
  


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
  print type(features)


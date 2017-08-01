
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

load_tableDatabase(spark, 'ReviewAnalysis_features')
load_tableDatabase(spark, 'ReviewAnalysis')

df= spark.sql("SELECT product, ID from ReviewAnalysis GROUP BY product")
df.show()

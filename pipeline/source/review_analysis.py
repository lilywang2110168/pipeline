
from spark import (get_sc, load_table)
import pyspark
import os
import time
import json

sc = get_sc()
spark = pyspark.sql.SparkSession(sc)

load_tableDB(spark, 'LilyLaptopReviews')
load_tableSQL()
df = spark.sql('SELECT reviewText from LilyLaptopReviews')

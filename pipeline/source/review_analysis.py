
from spark import (get_sc, load_table)
import pyspark
import os
import time
import json

sc = get_sc()
spark = pyspark.sql.SparkSession(sc)

load_table(spark, 'LilyLaptopReviews', 'ISC.DM.{}')
load_table(spark, 'Category', 'Database.{}')

df = spark.sql("SELECT features from Category WHERE categoryName ='Laptops'")
df.show()



from spark import (get_sc, load_tableDatabase, load_tableISCDM)
import pyspark
import os
import time
import json

sc = get_sc()
spark = pyspark.sql.SparkSession(sc)

load_tableISCDM(spark, 'LilyLaptopReviews')
load_tableDatabase(spark, 'Category')

df = sparkDatabase.sql("SELECT features, categoryName from Category")
df.show()


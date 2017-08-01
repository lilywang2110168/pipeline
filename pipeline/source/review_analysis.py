
import pyspark

from spark import (get_sc, load_table)

sc = get_sc()
spark = pyspark.sql.SparkSession(sc)
load_table(spark, 'Category')
df = spark.sql('SELECT one from natetest')
df.show()

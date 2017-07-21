import os
os.environ['PYSPARK_SUBMIT_ARGS'] = '--jars /Applications/Ensemble/dev/java/lib/JDK18/cache-jdbc-2.0.0.jar,/Applications/Ensemble/dev/java/lib/JDK18/cache-spark-2.0.0.jar pyspark-shell'
import pyspark

def get_sc():
    conf = pyspark.SparkConf() \
        .set('spark.cache.master.url', 'Cache://localhost:1972/USER') \
        .set("spark.cache.master.user", "_system") \
        .set("spark.cache.master.password", "SYS")
    return pyspark.SparkContext(master='local[*]', appName='reviews', conf=conf)

def load_table(spark, table_name):
    df = spark.read \
        .format('com.intersys.spark') \
        .option('dbtable', 'ISC_DM.{}'.format(table_name)) \
        .load()
    df.registerTempTable(table_name)
    spark.sql('CACHE TABLE {}'.format(table_name)).collect()
    return df

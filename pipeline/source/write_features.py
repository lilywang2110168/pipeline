from spark import (get_sc, load_table)
from pyspark.sql import Row
import pyspark

sc = get_sc()
spark = pyspark.sql.SparkSession(sc)

featureList=['electronics', 'laptops', 'heaphones']

rdd = sc.parallelize(featureList)
category = rdd.map(lambda x: Row(categoryName=x))
schemaCategory = spark.createDataFrame(category)
schemaCategory.show()



##this line of code overwrite a table!!
##schemaCategory.write.mode('overwrite').format('com.intersys.spark').option('dbtable', 'ISC_DM.{}'.format("testCategory")).save()

##this line of code appends to a table!
##schemaCategory.write.mode('append').format('com.intersys.spark').option('dbtable', 'ISC_DM.{}'.format("testCategory")).save()

schemaCategory.write.mode('append').format('com.intersys.spark').option('dbtable', 'Database.{}'.format("CategoryTest")).save()

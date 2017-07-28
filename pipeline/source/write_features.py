from spark import (get_sc, load_table)
from pyspark.sql import Row
import pyspark

sc = get_sc()
spark = pyspark.sql.SparkSession(sc)

featureList=[('machine', 38046), ('quality', 20615), ('software', 19452), ('model', 17034), ('program', 16186), ('window', 15888), ('size', 15770), ('graphic', 13272), ('speed', 13113), ('speaker', 13094), ('desktop', 12722), ('battery life', 18188), ('hard drive', 13866), ('touch screen', 9243), ('customer service', 5101), ('operating system', 4059), ('mouse pad', 2866), ('backlit keyboard', 2856), ('light weight', 2623), ('build quality', 2405), ('video card', 2031), ('tech support', 2026), ('optical drive', 1821), ('screen size', 1802), ('screen resolution', 1736), ('big deal', 1709), ('power cord', 1647), ('word processing', 1560), ('wireless mouse', 1519), ('sound quality', 1494), ('number pad', 1454), ('desktop replacement', 1435)]

rdd = sc.parallelize(featureList)
category = rdd.map(lambda x: Row(feature=x[0], category=int(x[1])))
schemaCategory = spark.createDataFrame(category)
schemaCategory.show()


schemaCategory.write.mode("overwrite").format('com.intersys.spark').option('dbtable', 'ISC_DM.{}'.format(testCategory))


from spark import (get_sc, load_table)
from pyspark.sql import Row
import pyspark

sc = get_sc()
spark = pyspark.sql.SparkSession(sc)

featureList=[('battery life', 18246), ('hard drive', 13871), ('touch screen', 9142), ('customer service', 5103), ('operating system', 4034), ('mouse pad', 2868), ('backlit keyboard', 2851), ('light weight', 2535), ('build quality', 2425), ('tech support', 2029), ('video card', 2027), ('optical drive', 1817), ('screen size', 1787), ('screen resolution', 1690), ('power cord', 1622), ('word processing', 1558), ('wireless mouse', 1516), ('number pad', 1455), ('desktop replacement', 1431), ('sound quality', 1423), ('power button', 1404), ('machine', 38026), ('quality', 20565), ('software', 19433), ('model', 17029), ('program', 16181), ('window', 15763), ('size', 15679), ('graphic', 13216), ('speaker', 13088), ('speed', 13049), ('desktop', 12682)]

rdd = sc.parallelize(featureList)
categoryFeature = rdd.map(lambda x: Row(featureName=x[0], popularityScore=x[1], sentimentScore=0))
CategoryFeature = spark.createDataFrame(categoryFeature)
CategoryFeature.show()


df = spark.createDataFrame(Row(ID='laptops', categoryName='laptops', feautures=categoryFeature))

df.show()
##this line of code overwrite a table!!
##schemaCategory.write.mode('overwrite').format('com.intersys.spark').option('dbtable', 'ISC_DM.{}'.format("testCategory")).save()

##this line of code appends to a table!
##schemaCategory.write.mode('append').format('com.intersys.spark').option('dbtable', 'ISC_DM.{}'.format("testCategory")).save()

##schemaCategory.write.mode('append').format('com.intersys.spark').option('dbtable', 'Database.{}'.format("CategoryTest")).save()

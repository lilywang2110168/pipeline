from spark import (get_sc, load_table)
from pyspark.sql import Row
import pyspark
import json

category ='Laptops'
myFile=open('features.txt', 'w')
featurelist=[('battery life', 13678), ('hard drive', 10898), ('touch screen', 5479), ('customer service', 3972), ('operating system', 2900), ('backlit keyboard', 2113), ('mouse pad', 2103), ('build quality', 1881), ('light weight', 1641), ('video card', 1631), ('tech support', 1611), ('optical drive', 1457), ('power cord', 1299), ('screen size', 1295), ('screen resolution', 1215), ('word processing', 1181), ('wireless mouse', 1145), ('desktop replacement', 1130), ('number pad', 1115), ('power button', 1070), ('sound quality', 1057), ('machine', 29086), ('software', 15176), ('program', 12285), ('size', 11460), ('port', 10916), ('window', 10831), ('graphic', 10357), ('speaker', 10275), ('button', 9890), ('desktop', 9632), ('speed', 9481)]

data = {}
data['categoryName'] = category
features=[]
for item in featurelist:
  feature={}
  feature["featureName"]=item[0]
  feature["popularityScore"]=item[1]
  features.append(feature)
  
data['features']=features
json.dump(data, myFile)
##this line of code overwrite a table!!
##schemaCategory.write.mode('overwrite').format('com.intersys.spark').option('dbtable', 'ISC_DM.{}'.format("testCategory")).save()

##this line of code appends to a table!
##schemaCategory.write.mode('append').format('com.intersys.spark').option('dbtable', 'ISC_DM.{}'.format("testCategory")).save()

##schemaCategory.write.mode('append').format('com.intersys.spark').option('dbtable', 'Database.{}'.format("CategoryTest")).save()

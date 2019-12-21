import pyspark
from pyspark import SparkContext
from pyspark.sql import SQLContext
sc =SparkContext()
sqlContext = SQLContext(sc)
from pyspark.sql.session import SparkSession
sc = SparkContext.getOrCreate()
spark = SparkSession(sc)
print('xxxxxxxxxxxxxxxxx -CONTEXT-GENERATOR-xxxxxxx #1')
from pyspark.sql import Row
from pyspark import SparkFiles
from pyspark.sql.types import *
from pyspark.ml import Pipeline
print('xxxxxxxxxxxxxxxxx -PIPELINE- xxxxxxxxxxxxxxx #2')

from pyspark.ml.feature import StringIndexer,OneHotEncoder, VectorAssembler,OneHotEncoderEstimator
from pyspark.sql.functions import col, countDistinct
import pyspark.sql.functions as f
from pyspark.sql.window import Window
from pyspark.sql.functions import rank, col
print('xxxxxxxxxxxxxxxxx -STRINGIndexer- xxxxxxxxxx #3')

url = "https://raw.githubusercontent.com/guru99-edu/R-Programming/master/adult_data.csv"
sc.addFile(url)
Dataf = sqlContext.read.csv(SparkFiles.get("adult_data.csv"), header=True, inferSchema= True)

print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx HEADS xxxxxxxxxxxx ')
windowX = Window.partitionBy(Dataf['income']).orderBy(Dataf['age'].desc())
Dataf.select('income','gender','workclass','education','educational-num','income', rank().over(windowX).alias('pemba')).filter(col('pemba') <= 50000).show() 
print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx GROUPLIST xxxxxxxx ')
Dataf.printSchema()
print((Dataf.count(), len(Dataf.columns)))
print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx COUNT xxxxxxxxxxx 3')

Dataf.select(*[f.collect_set(c).alias(c) for c in Dataf.columns]).show()
print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx DSTINCT xxxxxxxxx 4')

Dataf.describe().show()
print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx EXPLAIN xxxxxxxxx 5')

Dataf.crosstab('age', 'income').sort("age_income").show()
print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx SALARY xxxxxxxxxxx 6')

Dataf.select('age','workclass','education','educational-num','marital-status','race','gender','native-country','income').show()
print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx FREE-PREDICT xxxxx 7')

print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx DATA SCIENCE ONE #1 SUCCESSFUL xxxxxxxxxx DONE!')


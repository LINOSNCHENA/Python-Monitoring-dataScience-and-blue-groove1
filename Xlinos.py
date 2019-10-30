import pyspark
from pyspark import SparkContext
sc =SparkContext()
from pyspark.sql import Row
from pyspark.sql import SQLContext
from pyspark import SparkFiles
sqlContext = SQLContext(sc)
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
sc = SparkContext.getOrCreate()
spark = SparkSession(sc)
from pyspark.ml.linalg import DenseVector
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer,OneHotEncoder, VectorAssembler,OneHotEncoderEstimator
from pyspark.sql.functions import col, countDistinct
import pyspark.sql.functions as f
url = "https://raw.githubusercontent.com/guru99-edu/R-Programming/master/adult_data.csv"
from pyspark import SparkFiles
sc.addFile(url)
from pyspark.sql import *
import numpy as np
from pyspark.sql import SQLContext
from pyspark.sql.functions import lit
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
print('''xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx B1''')
# Create the userTable
UserStatus = Row("user_id", "name", "gender")
status1 = UserStatus(1,"A", 'M')
status2  = UserStatus(2,"B", 'F')
status3 = UserStatus(3,"C", 'F')
status4  = UserStatus(4,"D", 'M')
status5  = UserStatus(5,"E", 'F')
statusA=Row(usersTable=[status1])
statusB=Row(usersTable=[status2])
statusC=Row(usersTable=[status3])
statusD=Row(usersTable=[status4])
statusE=Row(usersTable=[status5])


# Create the transactionTable
times = Row("index", "user_id", "age","weight")
time1=times(1,1,23,60)
time2=times(2,2,34,75)
time3=times(3,3,55,80)
time4=times(4,3,43,90)
time5=times(5,4,54,66)
time6=times(6,7,23,57)
time7=times(7,4,65,64)
time8=times(8,7,44,80)
timeA=Row(transactionTable=[time1])
timeB=Row(transactionTable=[time2])
timeC=Row(transactionTable=[time3])
timeD=Row(transactionTable=[time4])
timeE=Row(transactionTable=[time5])
timeF=Row(transactionTable=[time6])
timeE1=Row(transactionTable=[time5])
timeF1=Row(transactionTable=[time6])

prodB = [statusA,statusB,statusC,statusD,statusE]
df6 = spark.createDataFrame(prodB)
prodC = [timeA,timeB,timeC,timeD,timeE,timeF,timeE1,timeF1]
df7 = spark.createDataFrame(prodC)

df6.select('usersTable.user_id','usersTable.name','usersTable.gender').show()
df7.select('transactionTable.index','transactionTable.user_id','transactionTable.age','transactionTable.weight').show()
print('''xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx B2''')
ta = df6.alias('ta')
tb = df7.alias('tb')
joinA = ta.join(tb, ta.usersTable.user_id <> tb.transactionTable.user_id,how='left')
joinB = ta.join(tb, ta.usersTable.user_id > tb.transactionTable.user_id,how='left')
joinA.show()
joinB.show()
print('''xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx B3''')






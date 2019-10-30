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
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{array, col, explode, lit, struct}


##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


Dataf= sqlContext.read.csv(SparkFiles.get("adult_data.csv"), header=True, inferSchema= True)

print('''xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx NEW-FEATURES xxxxxxxxxxxxxxxxxxxxx B1''')
# Create the Employees
Employee = Row("employeeID", "Name", "profession")
employee6 = Employee(6, "Mavin", 'IT')
employee1 = Employee(5, 'Nikolas', 'SALES')
employee2 = Employee(4, 'Leon', 'IT')
employee3 = Employee(3,'Lorena', 'IT')
employee4 = Employee(2, 'wendell', 'OTHERS')
employee5 = Employee(1, 'michael', 'IT')
# Create the EmployeesObject instances from Employees
EmployeesA = Row(employees=[employee1])
EmployeesB = Row(employees=[employee2])
EmployeesC = Row(employees=[employee3])
EmployeesD = Row(employees=[employee4])
EmployeesE = Row(employees=[employee5])
EmployeesF = Row(employees=[employee6])

# Create the userStatus
UserStatus = Row("userstatus", "Activity before 60 days", "activity 30-60 days ago","Activity in the last 30 days")
status1 = UserStatus("New", '-', '-','x')
status2  = UserStatus("Active", 'NA', 'x','x')
status3 = UserStatus("Churned", 'NA', 'x','-')
status4  = UserStatus("Reactivated", 'x', '-','x')
statusA=Row(state=[status1])
statusB=Row(state=[status2])
statusC=Row(state=[status3])
statusD=Row(state=[status3])

# Create the ActivityTable
times = Row("date", "user_id")
time1=times('','')
time2=times('','')
time3=times('','')
time4=times('','')
timeA=Row(time1=[time1])
timeB=Row(time1=[time2])
timeC=Row(time1=[time3])
timeD=Row(time1=[time4])

prodA = [EmployeesA,EmployeesB,EmployeesC,EmployeesD,EmployeesE,EmployeesF]
df3 = spark.createDataFrame(prodA)
prodB = [statusA,statusB,statusC,statusD]
df4 = spark.createDataFrame(prodB)
prodC = [timeA,timeB,timeC,timeD]
df5 = spark.createDataFrame(prodC)


df4.select('state.userstatus').show()
df4.select('state.Activity before 60 days').show()
df4.select('state.activity 30-60 days ago').show()
df4.select('state.Activity in the last 30 days').show()

dt = sc.parallelize([ (k,) + tuple(v[0:]) for k,v in df4.items()]).toDF()
dt.show()
    
#transpose(df4).show()
#df5.select('time1.date').show()
#df3.groupBy("employees.profession").count().sort("count",ascending=True).show()
print('''xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx B2''')






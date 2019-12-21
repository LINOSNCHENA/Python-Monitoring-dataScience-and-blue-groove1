import pyspark
from pyspark import SparkContext
sc =SparkContext()
from pyspark.sql import Row
from pyspark.sql import SQLContext
from pyspark import SparkFiles
sqlContext = SQLContext(sc)
#from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
sc = SparkContext.getOrCreate()
spark = SparkSession(sc)
from pyspark.sql import *
from pyspark.sql.types import *
#from pyspark.ml import Pipeline
import numpy as np
from pyspark.sql.functions import lit


print('''xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx STARTING''')
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

prodA = [EmployeesA,EmployeesB,EmployeesC,EmployeesD,EmployeesE,EmployeesF]
df3 = spark.createDataFrame(prodA)

df3.select('employees.employeeID','employees.Name','employees.profession').show()
df3.groupby('employees.profession').count().show()

print('''xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx 1''')

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

prodB = [statusA,statusB,statusC,statusD]
df4 = spark.createDataFrame(prodB)
prodC = [timeA,timeB,timeC,timeD]
df5 = spark.createDataFrame(prodC)

df4.select('state.userstatus','state.Activity before 60 days','state.activity 30-60 days ago','state.Activity in the last 30 days').show()
print('''xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx 3A''')

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
print('''xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx 3B''')

t1=df6.select('usersTable.user_id','usersTable.name','usersTable.gender')
t2=df7.select('transactionTable.index','transactionTable.user_id','transactionTable.age','transactionTable.weight')
ta = t1.alias('ta')
tb = t2.alias('tb')
joinA = ta.join(tb, ta.user_id < tb.user_id,how='left')
joinB = ta.join(tb, ta.user_id != tb.user_id,how='left')
joinA.show()
joinB.show()
print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx DATA SCIENCE THREE #3 SUCCESSFUL xxxxxxxxxx DONE!')






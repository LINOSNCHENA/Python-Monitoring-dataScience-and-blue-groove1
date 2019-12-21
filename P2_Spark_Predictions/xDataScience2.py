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

from pyspark.ml.feature import StringIndexer,OneHotEncoder, VectorAssembler, OneHotEncoder

from pyspark.sql.functions import col, countDistinct
import pyspark.sql.functions as f
from pyspark.sql.window import Window
from pyspark.sql.functions import rank, col
print('xxxxxxxxxxxxxxxxx -STRINGIndexer- xxxxxxxxxx #3')
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import DenseVector
from pyspark.ml.evaluation import BinaryClassificationEvaluator
print('xxxxxxxxxxxxxxxxx -TRANSFORMER- xxxxxxxxxxxx #4')

url = "https://raw.githubusercontent.com/guru99-edu/R-Programming/master/adult_data.csv"
sc.addFile(url)
Dataf = sqlContext.read.csv(SparkFiles.get("adult_data.csv"), header=True, inferSchema= True)

print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx TRANSFORM+BOTH xxxxxx 8')
number_FEATURES = ['age', 'x', 'fnlwgt', 'hours-per-week']
letter_FEATURES = ['workclass', 'education', 'marital-status', 'native-country']
stages = [] 
for categoricalCol in letter_FEATURES:
    stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "Choma")
    #encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()],outputCols=[categoricalCol + "Vasteras"])
    encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()],outputCols=[categoricalCol + "Vasteras"])
    stages += [stringIndexer, encoder]
# Convert income into lusaka using the StringIndexer
stringLusaked =  StringIndexer(inputCol="income", outputCol="lusaka")
stages += [stringLusaked]
assemblerInputs = [c + "Vasteras" for c in letter_FEATURES] + number_FEATURES
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="featureX")
stages += [assembler]
# Create a Pipeline.
pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(Dataf)
model = pipelineModel.transform(Dataf)
model.take(4)
model.show()
print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx NEW-FEATURE xxxxxxxxxxxxxxx 9')


df_model1 = model.rdd.map(lambda x: (x["lusaka"], DenseVector(x["featureX"])))
df_model2 = sqlContext.createDataFrame(df_model1, ["label", "featureX"])

train_data, test_data = df_model2.randomSplit([.8,.2],seed=1234)
train_data.groupby('label').agg({'label': 'count'}).show()
test_data.groupby('label').agg({'label': 'count'}).show()
LR = LogisticRegression(labelCol="label",
                        featuresCol="featureX",
                        maxIter=10,regParam=0.3)
linearModel = LR.fit(train_data)
predictions = linearModel.transform(test_data)
predictions.printSchema()
selectedPredictions = predictions.select("label", "prediction", "probability")
selectedPredictions.show(8)

print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx PREDICTIONS xxxxxxxxxxxxx 10')
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
print("PENCENTAGE OF GUESS ERROR :")
print(evaluator.evaluate(predictions))
print(evaluator.getMetricName())
print("TYPE OF TESTS CONDUCTED :"+evaluator.getMetricName())
print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx ROC METRIX xxxxxxxxxxxxx 11')
print("")
print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx DATA SCIENCE TWO #2 SUCCESSFUL xxxxxxxxxx DONE!')









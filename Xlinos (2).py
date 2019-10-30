import pyspark
from pyspark import SparkContext
sc =SparkContext()
from pyspark.sql import Row
from pyspark.sql import SQLContext
from pyspark import SparkFiles
sqlContext = SQLContext(sc)
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
sc = SparkContext.getOrCreate()
spark = SparkSession(sc)
from pyspark.ml.linalg import DenseVector
from pyspark.ml.feature import StringIndexer,OneHotEncoder, VectorAssembler,OneHotEncoderEstimator
from pyspark.sql.functions import col, countDistinct
import pyspark.sql.functions as f
url = "https://raw.githubusercontent.com/guru99-edu/R-Programming/master/adult_data.csv"
sc.addFile(url)
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
url = "https://raw.githubusercontent.com/guru99-edu/R-Programming/master/adult_data.csv"
from pyspark import SparkFiles
sc.addFile(url)
Dataf = sqlContext.read.csv(SparkFiles.get("adult_data.csv"), header=True, inferSchema= True)
print('''xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx3''')
Dataf.printSchema()
print((Dataf.count(), len(Dataf.columns)))
print('''xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxCOUNTxxxxxxxxxxx3''')
Dataf.select(*[f.collect_set(c).alias(c) for c in Dataf.columns]).show()
print('''xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxDSTINCTxxxxxxxxx4''')
Dataf.describe().show()
print('''xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxEXPLAINxxxxxxxxx5''')
Dataf.crosstab('age', 'income').sort("age_income").show()
print('''xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxSALARYxxxxxxxxxxx6''')
Dataf.select('age','workclass','education','educational-num','marital-status','race','gender','native-country','income').show()
print('''xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxFREEpREDICTIONxxxxxxxxxxx7''')
print('''xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxQUESTION-NOxxxxxxxxxxxx8''')
indexer = StringIndexer(inputCol="workclass", outputCol="workclass_index").fit(Dataf)
indexed = indexer.transform(Dataf)
encoder = OneHotEncoder(dropLast=False, inputCol="workclass_index", outputCol="workclass_vec")
encoded = encoder.transform(indexed)
CONTI_FEATURES = ['age', 'x', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
CATE_FEATURES = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']
stages = [] # stages in our Pipeline
for categoricalCol in CATE_FEATURES:
    stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "Index")
    encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()],outputCols=[categoricalCol + "classVec"])
    stages += [stringIndexer, encoder]
# Convert income into label indices using the StringIndexer
label_stringIdx =  StringIndexer(inputCol="income", outputCol="newlabel")
stages += [label_stringIdx]
assemblerInputs = [c + "classVec" for c in CATE_FEATURES] + CONTI_FEATURES
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]
# Create a Pipeline.
pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(Dataf)
model = pipelineModel.transform(Dataf)
model.take(5)
model.show()
print('''xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxNEWFeaturexxxxxxxxxxxxxxxxxxxxxxx 9''')
# Import `LinearRegression`
from pyspark.ml.classification import LogisticRegression

data2 = model.rdd.map(lambda x: (x["newlabel"], DenseVector(x["features"])))
df_train = sqlContext.createDataFrame(data2, ["label", "features"])
train_data, test_data = df_train.randomSplit([.8,.2],seed=1234)
train_data.groupby('label').agg({'label': 'count'}).show()
test_data.groupby('label').agg({'label': 'count'}).show()
LR = LogisticRegression(labelCol="label",
                        featuresCol="features",
                        maxIter=10,
                        regParam=0.3)
# Fit the data to the model
linearModel = LR.fit(train_data)
# Make predictions on test data using the transform() method.
predictions = linearModel.transform(test_data)
predictions.printSchema()
selected = predictions.select("label", "prediction", "probability")
selected.show(20)
print('''xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxPREDICTIONSxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx 10''')
### Use ROC 
from pyspark.ml.evaluation import BinaryClassificationEvaluator
# Evaluate model
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
print(evaluator.evaluate(predictions))
print("RESULTS FROM THE TESTING :"+ evaluator.getMetricName())
print('''xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxROC METRICxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx11''')
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++








# Databricks notebook source
import random
from pyspark.sql import *

print('hi')
print('bye2')

def generate_person():
  sex = 'male' if random.randint(0, 1) < 0.5 else 'female'
  score = random.randint(0, 10)
  salary = 20000 * score
  salary += 50000 if sex == 'male' else 30000
  return Row(sex=sex, score=score, salary=salary)

df = sqlContext.createDataFrame([generate_person() for i in range(0, 10)])
display(df)

# COMMAND ----------

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler

sexIndexer = StringIndexer(inputCol='sex', outputCol='indexedSex')
oneHotEncoder = OneHotEncoder(inputCol='indexedSex', outputCol='sexVector')
assembler = VectorAssembler(
    inputCols=['score', 'sexVector'],
    outputCol='features'
)
lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, labelCol='salary')
pipeline = Pipeline(stages=[sexIndexer, oneHotEncoder, assembler, lr])
model = pipeline.fit(df)
model.write().overwrite().save('test_model')

# COMMAND ----------

test_df = sqlContext.createDataFrame([generate_person() for i in range(0, 10)])
display(test_df)

loaded_model = PipelineModel.load('test_model')
display(loaded_model.transform(test_df))

# COMMAND ----------

dbutils.fs.ls('/test_model/')
print('hi')

# Databricks notebook source
#https://github.com/GdMacmillan/spark_recommender_systems/blob/master/spark_recommender.ipynb
#https://towardsdatascience.com/build-recommendation-system-with-pyspark-using-alternating-least-squares-als-matrix-factorisation-ebe1ad2e7679

# COMMAND ----------

!pip install lightfm;
!pip install mlflow;

# COMMAND ----------

# DBTITLE 1,Libraries
from pyspark.sql.functions import *
import pandas as pd
import numpy as np
from pyspark.sql.window import Window
from lightfm.datasets import fetch_movielens
from lightfm import LightFM
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# COMMAND ----------

dbutils.fs.mkdirs("recsys-als-lightfm");
dbutils.fs.ls("recsys-als-lightfm");

# COMMAND ----------

# MAGIC %sh
# MAGIC git clone https://github.com/VD-git/recsys-als-lightfm.git recsys-als-lightfm/

# COMMAND ----------

# DBTITLE 1,Importing Tables
links = spark.read.csv('file:/databricks/driver/recsys-als-lightfm/data-source/links.csv', header=True, inferSchema=True)
ratings = spark.read.csv('file:/databricks/driver/recsys-als-lightfm/data-source/ratings.csv', header=True, inferSchema=True)
tags = spark.read.csv('file:/databricks/driver/recsys-als-lightfm/data-source/tags.csv', header=True, inferSchema=True)
movies = spark.read.csv('file:/databricks/driver/recsys-als-lightfm/data-source/movies.csv', header=True, inferSchema=True)

# COMMAND ----------

SEED = 99

# COMMAND ----------

train, cross, test = ratings.randomSplit([0.70, 0.15, 0.15], seed = SEED)

# COMMAND ----------

als = ALS(userCol="userId", 
          itemCol="movieId",
          ratingCol="rating", 
          nonnegative = True, 
          implicitPrefs = False,
          coldStartStrategy="drop",
          seed=SEED
         )

# COMMAND ----------

# DBTITLE 1,Constructing GridSpec to Find Best Parameters
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

paramGrid = ParamGridBuilder() \
    .addGrid(als.regParam, [0.01, 0.1, 1]) \
    .addGrid(als.rank, np.linspace(start = 50, stop = 150, num = 3)) \
    .build()

crossval = CrossValidator(estimator=als,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=4)
cvModel = crossval.fit(train)

# COMMAND ----------

model_cv = {'regularization':[], 'rank_factor':[], 'rmse':[]}
for i, j in zip(range(len(paramGrid)), cvModel.avgMetrics):
  model_cv['regularization'].append(list(paramGrid[i].values())[0])
  model_cv['rank_factor'].append(list(paramGrid[i].values())[1])
  model_cv['rmse'].append(j)

# COMMAND ----------

pd.DataFrame(model_cv).sort_values(by='rmse').head()

# COMMAND ----------

regularization = cvModel.bestModel._java_obj.parent().getRegParam()
rank = cvModel.bestModel._java_obj.parent().getRank()

# COMMAND ----------

# DBTITLE 1,Training with the best Parameters (Finding maxIter)
rmse_error = {'iterations':[], 'rmse':[]}
for it in np.linspace(start = 1, stop = 10, num = 10):
  als = ALS(userCol="userId", 
          itemCol="movieId",
          ratingCol="rating",
          rank=rank,
          regParam=regularization,
          nonnegative = True, 
          implicitPrefs = False,
          coldStartStrategy="drop",
          maxIter=it,
          seed=SEED)
  model = als.fit(train)
  y_pred = model.transform(cross)
  rmse_error['iterations'].append(it)
  rmse_error['rmse'].append(evaluator.evaluate(y_pred))

# COMMAND ----------

pd.DataFrame(rmse_error).sort_values(by='rmse').head()

# COMMAND ----------

als = ALS(userCol="userId", 
          itemCol="movieId",
          ratingCol="rating",
          rank=rank,
          regParam=regularization,
          nonnegative = True, 
          implicitPrefs = False,
          coldStartStrategy="drop",
          maxIter=pd.DataFrame(rmse_error).sort_values(by='rmse').iloc[0,0],
          seed=SEED)
model = als.fit(train)

# COMMAND ----------

# DBTITLE 1,Recommendations by each User
table_recommendation = (model.recommendForAllUsers(5)
                             .withColumn('rec_1', col('recommendations').getItem(0).movieId)
                             .withColumn('rec_2', col('recommendations').getItem(1).movieId)
                             .withColumn('rec_3', col('recommendations').getItem(2).movieId)
                             .withColumn('rec_4', col('recommendations').getItem(3).movieId)
                             .withColumn('rec_5', col('recommendations').getItem(4).movieId)
                             .drop('recommendations')
                       )

table_recommendation_names = (table_recommendation.join(movies, on = [table_recommendation.rec_1 == movies.movieId], how = 'inner').withColumnRenamed('title','title_rec_1').drop('genres', 'rec_1', 'movieId')
                                                  .join(movies, on = [table_recommendation.rec_2 == movies.movieId], how = 'inner').withColumnRenamed('title','title_rec_2').drop('genres', 'rec_2', 'movieId')
                                                  .join(movies, on = [table_recommendation.rec_3 == movies.movieId], how = 'inner').withColumnRenamed('title','title_rec_3').drop('genres', 'rec_3', 'movieId')
                                                  .join(movies, on = [table_recommendation.rec_4 == movies.movieId], how = 'inner').withColumnRenamed('title','title_rec_4').drop('genres', 'rec_4', 'movieId')
                                                  .join(movies, on = [table_recommendation.rec_5 == movies.movieId], how = 'inner').withColumnRenamed('title','title_rec_5').drop('genres', 'rec_5', 'movieId')
                                                  .select('userId', 'title_rec_1', 'title_rec_2', 'title_rec_3', 'title_rec_4', 'title_rec_5')
                             )

# COMMAND ----------

display(table_recommendation_names)

# COMMAND ----------

train.select('userId').distinct().count()

# COMMAND ----------



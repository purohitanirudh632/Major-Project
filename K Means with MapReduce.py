#!/usr/bin/env python
# coding: utf-8

# In[14]:


import os
import findspark
findspark.init()

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("PySpark").getOrCreate()


# In[15]:


# Initialize Spark session
spark = SparkSession.builder.appName("KMeansClusteringMapReduce").getOrCreate()


# In[16]:


import os
import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import time
from sklearn.datasets import fetch_20newsgroups


# In[17]:


# Load the 20 Newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))


# In[18]:


# Define schema for the DataFrame
schema = StructType([
    StructField("text", StringType(), True),
    StructField("category", IntegerType(), True)
])


# In[19]:


# Convert 20 Newsgroups dataset to a DataFrame
newsgroups_data = [(text, int(category)) for text, category in zip(newsgroups.data, newsgroups.target)]
df = spark.createDataFrame(newsgroups_data, schema=schema)


# In[20]:


# Tokenize the text column
tokenizer = Tokenizer(inputCol="text", outputCol="words")
wordsData = tokenizer.transform(df)


# In[21]:


# Text processing
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20000)
featurizedData = hashingTF.transform(wordsData)
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)


# In[22]:


# Apply K-means clustering with MapReduce
k = 5
num_iterations = 15
start_time = time.time()
kmeans = KMeans().setK(k).setSeed(42).setMaxIter(num_iterations).setFeaturesCol("features")
model = kmeans.fit(rescaledData)
predictions = model.transform(rescaledData)
mapreduce_duration = time.time() - start_time


# In[23]:


# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)
print(f"Silhouette Score: {silhouette}")
print(f"Time duration for K-means with MapReduce: {mapreduce_duration:.4f} seconds")


# In[24]:


# Show cluster sizes
cluster_counts = predictions.groupBy("prediction").count().collect()
for row in cluster_counts:
    print(f"Cluster {row['prediction']}: {row['count']} records")


# In[25]:


# Show a few documents from each cluster
for cluster_id in range(k):
    print(f"\nCluster {cluster_id} documents:")
    cluster_docs = predictions.filter(predictions.prediction == cluster_id).select("text").take(3)
    for doc in cluster_docs:
        print(doc.text)


# In[26]:


# Show the number of records in each cluster for the first cluster (Cluster 0)
cluster_0_counts = predictions.filter(predictions.prediction == 0).groupBy("category").count().collect()
categories = [newsgroups.target_names[row['category']] for row in cluster_0_counts]
counts = [row['count'] for row in cluster_0_counts]


# In[27]:


# Plot the graph
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
bars = plt.bar(categories, counts, color='skyblue')
plt.xlabel('Newsgroups')
plt.ylabel('Number of Records in Cluster 0')
plt.title('Number of Records in Cluster 0 Across Newsgroups')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability


# Annotate each bar with its value
for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, count,
             ha='center', va='bottom')

plt.tight_layout()
plt.show()

spark.stop()



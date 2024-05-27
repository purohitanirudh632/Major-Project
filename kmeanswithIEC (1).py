#!/usr/bin/env python
# coding: utf-8

# In[3]:


import time
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_20newsgroups
import numpy as np


# In[4]:


# Load the 20 Newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))


# In[11]:


# Convert text data to TF-IDF matrix
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(newsgroups.data)

# # Convert TF-IDF matrix to dense format for silhouette_score and davies_bouldin_score
# tfidf_matrix_dense = tfidf_matrix.toarray()


# In[12]:



# Specify the number of iterations (n_init parameter)
num_iterations = 5  # You can adjust this based on your preference


# Apply K-means clustering with k=5 and k-means++ initialization
start_time = time.time()
kmeans_kmeanspp = KMeans(n_clusters=5, init='k-means++',n_init=num_iterations,random_state=42)
clusters_kmeanspp = kmeans_kmeanspp.fit_predict(tfidf_matrix_dense)
kmeanspp_duration = time.time() - start_time

# # Calculate Silhouette Score and Davies-Bouldin Index
# silhouette_avg = silhouette_score(tfidf_matrix, clusters_kmeanspp)
# davies_bouldin_avg = davies_bouldin_score(tfidf_matrix, clusters_kmeanspp)

# print(f"Silhouette Score: {silhouette_avg}")
# print(f"Davies-Bouldin Index: {davies_bouldin_avg}")


        


# In[7]:


# Create a dictionary to store the counts for each cluster
cluster_counts = {i: np.sum(clusters_kmeanspp == i) for i in range(5)}

# Display the results
for cluster_id, count in cluster_counts.items():
    print(f"\nCluster {cluster_id + 1}: {count} records")
    
    # Count records from each original newsgroup within the cluster
    newsgroup_counts = {newsgroup_name: np.sum((clusters_kmeanspp == cluster_id) & (newsgroups.target == label))
                        for label, newsgroup_name in enumerate(newsgroups.target_names)}
    
    # Print the counts for each newsgroup within the cluster
    for newsgroup, newsgroup_count in newsgroup_counts.items():
        print(f"- {newsgroup}: {newsgroup_count} records")
        


# In[8]:


print(f"Time duration for K-means with k-means++ initialization: {kmeanspp_duration:.4f} seconds")


# In[9]:


# Get the number of records for cluster 1
cluster_1_counts = []
for label, newsgroup_name in enumerate(newsgroups.target_names):
    cluster_1_count = np.sum((clusters_kmeanspp == 0) & (newsgroups.target == label))
    cluster_1_counts.append(cluster_1_count)

# Plot the graph
plt.figure(figsize=(10, 6))
bars = plt.bar(newsgroups.target_names, cluster_1_counts, color='skyblue')
plt.xlabel('Newsgroups')
plt.ylabel('Number of Records in Cluster 1')
plt.title('Number of Records in Cluster 1 Across Newsgroups')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability

# Annotate each bar with its value
for bar, count in zip(bars, cluster_1_counts):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, count,
             ha='center', va='bottom')

plt.tight_layout()
plt.show()


# In[ ]:





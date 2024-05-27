#!/usr/bin/env python
# coding: utf-8

# In[2]:


import time
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.datasets import fetch_20newsgroups
import numpy as np

# Load the 20 Newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

# Convert text data to TF-IDF matrix
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(newsgroups.data)

#fix the iteration in this clusters
num_iterations = 15

# Apply K-means clustering with k=5
start_time = time.time()
kmeans_normal = KMeans(n_clusters=5,n_init =  num_iterations,random_state=42)
clusters = kmeans_normal.fit_predict(tfidf_matrix)
normal_duration = time.time() - start_time


# Create a dictionary to store the counts for each cluster
cluster_counts = {i: np.sum(clusters == i) for i in range(5)}

# Display the results
for cluster_id, count in cluster_counts.items():
    print(f"Cluster {cluster_id + 1}: {count} records")

# Display a few documents from each cluster
for cluster_id in range(5):
    print(f"\nCluster {cluster_id + 1} documents:")
    cluster_indices = np.where(clusters == cluster_id)[0]
    for idx in cluster_indices[:3]:  # Display the first 3 documents from each cluster
        print(f"Document {idx + 1} - {newsgroups.target_names[newsgroups.target[idx]]}:\n{newsgroups.data[idx]}\n")

# Display the results
for cluster_id, count in cluster_counts.items():
    print(f"\nCluster {cluster_id + 1}: {count} records")
    
    # Count records from each original newsgroup within the cluster
    newsgroup_counts = {newsgroup_name: np.sum((clusters == cluster_id) & (newsgroups.target == label))
                        for label, newsgroup_name in enumerate(newsgroups.target_names)}
    
    # Print the counts for each newsgroup within the cluster
    for newsgroup, newsgroup_count in newsgroup_counts.items():
        print(f"- {newsgroup}: {newsgroup_count} records")        
        


# In[ ]:





# In[18]:


print(f"Time duration for normal K-means: {normal_duration:.4f} seconds")


# In[13]:


# Apply K-means clustering with k=5 and fixed number of iterations
start_time = time.time()
kmeans_fixed_iter = KMeans(n_clusters=5, n_init=num_iterations, random_state=42)
clusters_fixed_iter = kmeans_fixed_iter.fit_predict(tfidf_matrix)
fixed_iter_duration = time.time() - start_time

# Get the number of records for cluster 1
cluster_1_counts_fixed_iter = []
for label, newsgroup_name in enumerate(newsgroups.target_names):
    cluster_1_count = np.sum((clusters_fixed_iter == 0) & (newsgroups.target == label))
    cluster_1_counts_fixed_iter.append(cluster_1_count)

# Plot the graph
plt.figure(figsize=(10, 6))
bars = plt.bar(newsgroups.target_names, cluster_1_counts_fixed_iter, color='skyblue')
plt.xlabel('Newsgroups')
plt.ylabel('Number of Records in Cluster 1')
plt.title(f'Number of Records in Cluster 1 Across Newsgroups (Fixed Iterations: {num_iterations})')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability

# Annotate each bar with its value
for bar, count in zip(bars, cluster_1_counts_fixed_iter):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, count,
             ha='center', va='bottom')

plt.tight_layout()
plt.show()


# In[4]:


# Count records from each original newsgroup across all clusters
newsgroup_records = {newsgroup: np.sum(newsgroups.target == label) for label, newsgroup in enumerate(newsgroups.target_names)}

# Display the counts for each newsgroup
for newsgroup, count in newsgroup_records.items():
    print(f"{newsgroup}: {count} records")


# In[3]:





# In[ ]:





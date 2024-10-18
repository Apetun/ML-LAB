#!/usr/bin/env python
# coding: utf-8

# ### Question 1

# ### Importing Libraries

# In[103]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display


# ### Initial 1D data

# In[104]:


# Initial Data Points
data = np.array([18, 22, 25, 27, 42, 43]).reshape(-1, 1)


# ### Initial proximity matrix

# In[105]:


# Initialize the proximity matrix (distance matrix) using Euclidean distance
n = len(data)
proximity_matrix = np.zeros((n, n))

# Compute the initial proximity matrix (distance matrix)
for i in range(n):
    for j in range(n):
        proximity_matrix[i, j] = np.abs(data[i] - data[j])
        
print(proximity_matrix)


# ### Agglomerative hierarchical clustering using single linkage

# In[106]:


# Function to print the proximity matrix
def print_proximity_matrix(matrix, clusters, iteration):
    print(f"\nProximity Matrix at Iteration {iteration}:")
    matrix[np.isinf(matrix)] = 0 
    # Create a new DataFrame for displaying clusters
    df = pd.DataFrame(matrix, columns=[str(clusters[i]) for i in range(len(clusters))], 
                        index=[str(clusters[i]) for i in range(len(clusters))])
    display(df)

# Agglomerative hierarchical clustering using Single Linkage (min distance)
clusters = [[data[i][0]] for i in range(n)]  # Initialize each point as a separate cluster

iteration = 0
print_proximity_matrix(proximity_matrix, clusters, iteration)

while len(clusters) > 1:
    # Find the two closest clusters
    min_dist = float('inf')
    to_merge = (None, None)

    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            # Calculate the single linkage distance (minimum distance between clusters)
            dist = np.min([np.abs(p1 - p2) 
                           for p1 in clusters[i] for p2 in clusters[j]])
            if dist < min_dist:
                min_dist = dist
                to_merge = (i, j)

    # Merge the two closest clusters
    i, j = to_merge
    clusters[i].extend(clusters[j])
    del clusters[j]

    # Update the proximity matrix
    new_matrix = np.full((len(clusters), len(clusters)), np.inf)
    for x in range(len(clusters)):
        for y in range(x + 1, len(clusters)):
            new_matrix[x, y] = new_matrix[y, x] = np.min([
                np.abs(p1 - p2)
                for p1 in clusters[x] for p2 in clusters[y]
            ])

    iteration += 1
    print_proximity_matrix(new_matrix, clusters, iteration)


# ### Plotting dendrogram using Scipy

# In[107]:


from scipy.cluster.hierarchy import dendrogram, linkage

# Plot the dendrogram using scipy
linkage_matrix = linkage(data, method='single')
plt.figure(figsize=(10, 5))
dendrogram(linkage_matrix, labels=[str(x) for x in data.flatten()])
plt.title('Dendrogram - Agglomerative Hierarchical Clustering (Single Linkage)')
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.show()


# ### Question 2

# ### Initial 2D data

# In[108]:


# Initial 2D Data Points
data = np.array([[1, 1], [3, 2], [9, 1], [3, 7], [7, 2], 
                   [9, 7], [4, 8], [8, 3], [1, 4]])


# ### Plotting Data

# In[112]:


# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], color='blue', s=100)  # 's' controls the size of the points
plt.title('Scatter Plot of Initial 2D Data Points')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.show()


# ### Agglomerative hierarchical clustering using single linkage

# In[109]:


# Function to print the proximity matrix
def print_proximity_matrix(matrix, iteration):
    print(f"\nProximity Matrix at Iteration {iteration}:")
    matrix[np.isinf(matrix)] = 0 
    df = pd.DataFrame(matrix, columns=[f"P{i}" for i in range(len(matrix))], 
                      index=[f"P{i}" for i in range(len(matrix))])
    display(df)

# Agglomerative hierarchical clustering using Single Linkage (min distance)
clusters = [[i] for i in range(n)]  # Initialize each point as a separate cluster

iteration = 0
print_proximity_matrix(proximity_matrix, iteration)

while len(clusters) > 1:
    # Find the two closest clusters
    min_dist = float('inf')
    to_merge = (None, None)
    
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            # Calculate the single linkage distance (minimum distance between clusters)
            dist = np.min([np.linalg.norm(data[p1] - data[p2]) 
                           for p1 in clusters[i] for p2 in clusters[j]])
            if dist < min_dist:
                min_dist = dist
                to_merge = (i, j)
    
    # Merge the two closest clusters
    i, j = to_merge
    clusters[i].extend(clusters[j])
    del clusters[j]
    
    # Update the proximity matrix
    new_matrix = np.full((len(clusters), len(clusters)), np.inf)
    for x in range(len(clusters)):
        for y in range(x + 1, len(clusters)):
            new_matrix[x, y] = new_matrix[y, x] = np.min([
                np.linalg.norm(data[p1] - data[p2])
                for p1 in clusters[x] for p2 in clusters[y]
            ])
    
    iteration += 1
    print_proximity_matrix(new_matrix, iteration)


# ### Initial proximity matrix

# In[110]:


# Initialize the proximity matrix (distance matrix) using Euclidean distance
n = len(data)
proximity_matrix = np.zeros((n, n))

# Compute the initial proximity matrix (Euclidean distance)
for i in range(n):
    for j in range(n):
        proximity_matrix[i, j] = np.linalg.norm(data[i] - data[j])
        
df = pd.DataFrame(proximity_matrix, columns=[f"P{i}" for i in range(len(proximity_matrix))],
                  index=[f"P{i}" for i in range(len(proximity_matrix))])
display(df)


# ### Plotting dendrogram using Scipy

# In[111]:


linkage_matrix = linkage(data, method='single')
plt.figure(figsize=(10, 5))
dendrogram(linkage_matrix, labels=[str(x) for x in range(len(data))])
plt.title('Dendrogram - Agglomerative Hierarchical Clustering (Single Linkage)')
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.show()


# In[ ]:





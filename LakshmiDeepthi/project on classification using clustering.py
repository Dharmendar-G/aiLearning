#!/usr/bin/env python
# coding: utf-8

# ### Task
# - predict the type of the class , clustering the dataset using clustering techniques.

# ### Importing the necessary  modules

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')


# ### importing and exporting file from AwS s3

# In[2]:


import boto3

s3 = boto3.resource("s3")
# Print out bucket names
for bucket in s3.buckets.all():
    print(bucket.name)
# uploading the file
# s3.upload_file(
#     Filename="C:\\Users\\LakshmiDeepthiGadams\\Desktop\\AWS_S3\\seeds_dataset_copy_s3.txt",
#     Bucket="deepu-1896",
#     Key="seeds_dataset_copy_s3.txt",
# )
# downloading the file

s3 = boto3.client("s3")
s3.download_file(
    Bucket="deepu-1896", # give name of the bucket
    Key="seeds_dataset.csv", #name of your file with extension
    Filename="C:\\Users\\LakshmiDeepthiGadams\\Desktop\\python materials\ASSIGNMENTS_BOURNTEC\\project on classification\\seeds_dataset_copy_s3.csv") # give the location to save your file with extensio


# In[3]:


df=pd.read_csv("seeds_dataset_copy_s3.csv").drop('ID',axis=1)
df.head()


# ### Pre-processing

# In[4]:


df.info()


# ## Visualizations

# #### Univariate analysis

# In[5]:


def bar(x, y,**kwargs):
    fig = plt.figure()
    sns.barplot(x, y,**kwargs)
    return fig
bar(df['seedType'],df['widthOfKernel'])
bar(df['seedType'],df['perimeter'])
bar(df['seedType'],df['compactness'])
bar(df['seedType'],df['area'])
bar(df['seedType'],df['lengthOfKernel'])
bar(df['seedType'],df['asymmetryCoefficient'])
bar(df['seedType'],df['lengthOfKernelGroove'])
plt.show()


# #### Multivariate Analysis

# In[6]:


plt.figure(figsize=(16,8))
sns.pairplot(df,hue='seedType')


# In[7]:


x=df.drop('seedType',axis=1)
plt.figure(figsize=(10,8))
sns.heatmap(x[x.describe().columns.tolist()].corr(), cmap="YlGnBu", annot=True, xticklabels='auto')
plt.title('Correlation Between Numerical Features')
plt.show()


# In[10]:


x.head()


# ### Clustering Techniques

# In[11]:


from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN,AgglomerativeClustering


# #### KMEANS CLUSTERING

# In[12]:


kmeans=KMeans(n_clusters = 3,                 # Set amount of clusters
                init = 'k-means++',             # Initialization method for kmeans
                max_iter = 300,                 # Maximum number of iterations
                n_init = 10,                    # Choose how often algorithm will run with different centroid
                random_state = 0)               # Choose random state for reproducibility)


# In[13]:


kmeans.fit(x)


# In[14]:


kmeans.cluster_centers_


# In[18]:


labels = pairwise_distances_argmin(x, kmeans.cluster_centers_)
metrics = {"Model":kmeans, "Silhouette Score":silhouette_score(x,labels)}
print(f"Model: {kmeans}")
print(f"\nSilhouette coefficient: {silhouette_score(x,labels):0.2f}")
print(f"\n\nInertia:{kmeans.inertia_}")
print(f"\n\nLabels : {labels}")


# 
# **Intertia :**
# 
# Inertia measures how well a dataset was clustered by K-Means. It is calculated by measuring the distance between each data point and its centroid, squaring this distance, and summing these squares across one cluster. A good model is one with low inertia AND a low number of clusters ( K )
# 
# **Silhouette :**
# 
# Silhouette Coefficient or silhouette score is a metric used to calculate the goodness of a clustering technique. Its value ranges from -1 to 1.
# * score close to 1: Means clusters are well apart from each other and clearly distinguished.
# * score close to 0: Means clusters are indifferent, or we can say that the distance between clusters is not significant.
# * score close to -1: Means clusters are assigned in the wrong way

# ### Elbow method with automatic clusters

# In[19]:


wcss=[]
for i in range(1, 11):
    model = KMeans(n_clusters = i,     
                    init = 'k-means++',                 # Initialization method for kmeans
                    max_iter = 300,                     # Maximum number of iterations 
                    n_init = 10,                        # Choose how often algorithm will run with different centroid 
                    random_state = 0)                   # Choose random state for reproducibility
    model.fit(x)                              
    wcss.append(model.inertia_)
    
# Show Elbow plot
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')                               # Set plot title
plt.xlabel('Number of clusters')                        # Set x axis name
plt.ylabel('Within Cluster Sum of Squares (WCSS)')      # Set y axis name
plt.show()


# In[20]:


from kneed import KneeLocator
kl = KneeLocator( range(1, 11), wcss, curve="convex", direction="decreasing")
knee_point = kl.knee #elbow_point = kneedle.elbow
print(f'the location of Kneepoint is: { knee_point}') #print('Elbow: ', elbow_point)
kl.plot_knee()


# ### FUNCTION FOR APPLYING DIFFERENT CLUSTERING ALGORITHMS

# In[23]:


def apply_clustering(algorithm=None,n_clusters=3,data=None,scale=None, feat_selection=None):
    # Applying Model
    if algorithm in [KMeans, MiniBatchKMeans]:
        if algorithm == KMeans:
            model = algorithm(n_clusters=n_clusters)
            model.fit(data)
            cluster_centers = np.sort(model.cluster_centers_, axis = 0)
            labels = pairwise_distances_argmin(data, cluster_centers)
            metrics = {"Model": model, "Silhouette Score":silhouette_score(data,labels)}
        elif algorithm == MiniBatchKMeans:
            model = MiniBatchKMeans(init ='k-means++', n_clusters=n_clusters, batch_size=2, n_init=10, max_no_improvement=10, verbose=0)
            model.fit(data)
            labels = pairwise_distances_argmin(data, model.cluster_centers_)
            metrics = {"Model": model, "Silhouette Score":silhouette_score(data,labels)}
        print(f"Model: {model}")
        print(f"\nSilhouette coefficient: {silhouette_score(data,labels):0.2f}")
        print(f"\n\nInertia:{model.inertia_}")
        print(f"\n\nLabels : {labels}")
        return metrics
    elif algorithm in [DBSCAN, AgglomerativeClustering]:
        if algorithm == DBSCAN:
            model = algorithm(eps = 0.0375, n_clusters = 3)
            model.fit(data)
            metrics = {"Model": str(algorithm()), "Silhouette Score":silhouette_score(data, model.labels_)}
        elif algorithm == AgglomerativeClustering:
            model = algorithm(affinity='euclidean', linkage='ward')  
            model.fit_predict(data)
            metrics = {"Model": str(algorithm()), "Silhouette Score":silhouette_score(data, model.labels_)}
        print(f"Model: {model}")
        print(f"\nSilhouette coefficient: {silhouette_score(data, model.labels_)}")
        print(f"\nLabels: {model.labels_}")
        return metrics


# ### MINI BATCH KMEANS

# In[24]:


apply_clustering(algorithm=MiniBatchKMeans, n_clusters=3, data=x, feat_selection=True)


# ### AgglomerativeClustering

# In[127]:


k=[2,3,4,5]
for i in k:
    ac4 = AgglomerativeClustering(n_clusters=i )
    plt.figure(figsize =(6, 6))
    plt.scatter(x['perimeter'],x['widthOfKernel'],
                c = ac4.fit_predict(x), cmap ='rainbow')
    plt.show()


# In[26]:


apply_clustering(algorithm=AgglomerativeClustering, n_clusters=3, data=x, feat_selection=True)


# ### Best MODEL

# In[110]:


models = [KMeans, MiniBatchKMeans, AgglomerativeClustering]
metrics_df = pd.DataFrame()
for m in models: 
    print("--"*50)
    metrics = apply_clustering(algorithm=m, n_clusters=3, data=x, feat_selection=True)
    metrics_df = metrics_df.append(metrics, ignore_index=True)  
print(metrics_df)
# Silhouette Score Plot
metrics_df['Silhouette Score'].plot(figsize=(8,5))
plt.title('Silhouette Score for Clustering Models')
plt.show()


# 
# -From the above plot we observe **KMeans** is best model

# ### Silhouette coefficient for different clusters

# In[133]:


k=[2,3,4,5,6,7]
for i in k:   
    model= [KMeans(n_clusters=i), MiniBatchKMeans(n_clusters=i), AgglomerativeClustering(n_clusters=i)]
    score=[]
    for m in model:
        score.append(silhouette_score(x,m.fit_predict(x)))
    print(f'silhoutte score for {i} is:{score}')


# In[ ]:





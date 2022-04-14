#!/usr/bin/env python
# coding: utf-8

# #### <span style='color:gray'>Dharmendra's Notebook (Team 4)</span>

# ## Objective: <span style='color:orange'>Implement a model to classify insurance & non insurance pdf documents using clustering/classification Algorithms</span>

# ## <span style='color:green'>Project Domain - Insurance Sector</span>
# 
# #### Context : To classify insurance and non insurance pdf documents using Clustering/Classification Algorithms
# 
# #### Data : Insurance & non insurance pdf documents
# 
# #### Attribute information : To implement algorithms, features are extracted from the insurance documents using different Regular Expressions, NLP & OpenCV techniques. 

# # INDEX<a id="index"></a>
# 
# - ### i. Libraries & Packages
# -  1.[**Libraries**](#section_L)
# 
# - ### ii. Visualizations
# - <span style='color:blue'>**E**</span><span style='color:red'>**D**</span><span style='color:green'>**A :**</span> [*@Decorators*](#D)
# -  1.[**Flow Diagram**](#p1)
# -  2.[**Feature Extraction wordcloud**](#p2)
# -  3.[**Correlation Plot**](#p3)
# - **Plots for Clustering**
# -  1.[**Elbow Plot**](#p4)
# -  2.[**Elbow Knee Locator Plot**](#p5)
# -  3.[**Dendogram Plot**](#p6)
# -  4.[**Scatter Plot**](#p7)
# -  5.[**Silhoutte Visualization**](#p8)
# -  6.[**Best Model**](#BM)
# 
# - **Plots for Classification**
# -  1.[**Count Plot**](#p9)
# -  2.[**KNN Performance plot**](#p10)
# -  3.[**SVM Confusion Matrix Plot**](#p11)
# -  4.[**Decision Tree Plot**](#p12)
# -  5.[**Random Forest Plot**](#p13)
# 
# - ### iii. Data PreProcessing
# -  1.[**Preprocessing of Data**](#section_PD)
# -  2.[**Preprocessing using NLP**](#section_NLP)
# -  3.[**Feature Extraction**](#section_FE)
# -  4.[**Creating DataFrame**](#section_CD)
# -  5.[**Feature Selection**](#section_FSE)
# -  6.[**Feature Scaling**](#section_FSC)
# 
# 
# - ### iv. Modelling
# - **Clustering Algorithms**
# -  1.[**Clustering Techniques**](#section_CT)
# -  2.[**Mini Batch K Means**](#section_MBKM)
# -  3.[**DBSCAN**](#section_DB)
# -  4.[**Agglomerative Clustering**](#section_A)
# - **Classification Algorithms**
# -  1.[**KNN**](#section_KNN)
# -  2.[**Logistic Regression**](#section_LR)
# -  3.[**Support Vector Machine**](#section_SVM)
# -  4.[**Decision Tree**](#section_DT)
# -  5.[**Random Forest**](#section_RF)
# 
# - ### v. Learnings & References
# -  1.[**Learnings**](#section_LO)
# -  2.[**References**](#section_REF)
# 
# 
# 

# ## Flow Diagram (Steps & tasks)<a id="p1"></a>
# ![flowdiagram.png](attachment:flowdiagram.png)

# ### Import Relevant Libraries<a id="section_L"></a>

# In[1]:


# Importing required libraries
import os 
import re
import time
import nltk
import hashlib
import pandas as pd 
import numpy as np
import tracemalloc
from string import *
from PIL import Image
import seaborn as sns
from PIL import Image
from wordcloud import WordCloud
import plotly.express as px
from collections import Counter
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.datasets import make_blobs
from pdf2image import convert_from_path
from sklearn.metrics import silhouette_samples, silhouette_score
from nltk.tokenize import word_tokenize,ToktokTokenizer
plt.style.use('fivethirtyeight')
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\DharmendraGa_5wskc\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'


# #### [Index](#index)

# ### Decorators <a id="D"></a>

# In[2]:


# wordcloud plot
def wordcloud(plot):
    def figure():
        plot()
        t = " ".join(words)
        wordcloud = WordCloud(width=800,height=500,random_state=21,max_font_size=110).generate(t)
        plt.figure(figsize=(10,7))
        plt.imshow(wordcloud,interpolation = 'bilinear')
        plt.axis("off")
        plt.show()
    return figure

# Correlation heatmap
def heatmap(plot):
    def figure():
        plot()
        plt.figure(figsize=(10,8))
        sns.heatmap(df[df.describe().columns.tolist()].corr(), cmap="YlGnBu", annot=True, xticklabels='auto')
        plt.title('Correlation Between Numerical Features')
        plt.show()
    return figure

# Elbow plot
def elbowplot(plot):
    def figure():
        plot()
        plt.figure(figsize=(6,4))
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('Values of K')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method using Distortion')
        plt.show()
    return figure

# Elbow Knee Locator
def elbowKneeLocator(plot):
    def figure():
        plot()
        kl = KneeLocator( range(1, 6), inertias, curve="convex", direction="decreasing")
        knee_point = kl.knee #elbow_point = kneedle.elbow
        print('Knee: ', knee_point) #print('Elbow: ', elbow_point)
        kl.plot_knee()
        plt.show()
    return figure

# Scatter Plot
def scatterplot(plot):
    def figure():
        plot()
        plt.figure(figsize=(7, 5))  
        plt.scatter(df['insurance'],df[z], c=kmeanModel.labels_, s=150)
        plt.title('Kmeans Predicted Labels')
        plt.show()
    return figure

# Silhouette score plot decorator
def silhouetteplot(plot):
    def figure():
        plot()
        cols = feature_selection(data=df)
        X, y = make_blobs(
            n_samples=50,
            n_features=1,
            centers=4,
            cluster_std=1,
            center_box=(-10.0, 10.0),
            shuffle=True,
            random_state=1) # For reproducibility 

        range_n_clusters = [2, 3, 4, 5, 6]

        for n_clusters in range_n_clusters:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)
            ax1.set_xlim([-0.1, 1])
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(df[cols])
            silhouette_avg = silhouette_score(df[cols], cluster_labels)
            print("For n_clusters =",n_clusters,"The average silhouette_score is :",silhouette_avg)
            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(df[cols], cluster_labels)
            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
                ith_cluster_silhouette_values.sort()
                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i
                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(np.arange(y_lower, y_upper),0,ith_cluster_silhouette_values,facecolor=color,edgecolor=color,alpha=0.7)
                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")
            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            ax2.scatter(df[cols].iloc[:, 0], df[cols].iloc[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k")
            # Labeling the clusters
            centers = clusterer.cluster_centers_
            # Draw white circles at cluster centers
            ax2.scatter(centers[:, 0],centers[:, 1],marker="o",c="white",alpha=1,s=600,
                edgecolor="k")
            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")
            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")
            plt.suptitle("Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"% n_clusters,fontsize=14,fontweight="bold",)
        plt.show();
    return figure

# Confusion Matrix
def confusionMatrix(plot):
    def figure():
        plot()
        cm = confusion_matrix(y_test,predictions)
        plot_confusion_matrix(model,X_test,y_test)
        plt.title("Confusion Matrix")
        plt.show()
    return figure

# Decision Tree plot
def decisionTree(plot):
    def figure():
        plot()
        fig = plt.figure(figsize=(10,5))
        dt = plot_tree(treeClassifier,
                       feature_names=data.columns,
                       class_names=["0","1"],
                       filled=True,
                       precision=4,rounded=True)
        plt.show()
    return figure

# Random forest plot
def randomForestTree(plot):
    def figure():
        plot()
        fn=data.columns
        cn=['0','1']
        fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (1,1), dpi=400)
        tree=plot_tree(rfc.estimators_[0],
                       feature_names = fn, 
                       class_names=cn,
                       filled = True)
        plt.show()
    return figure


# ### Data folder

# In[3]:


# Files in the directory
os.chdir('C:\\Users\\DharmendraGa_5wskc\\Downloads\\pdf_data\\data')

# Total PDF Documents
pdf_list = []
for (root,dirs,file) in os.walk(os.getcwd()):
    for f in file:
        if ".pdf" in f:
            pdf_list.append(f)
            
print(pdf_list)
print("\nToTal No. of Pdfs : ",len(pdf_list))


# ## Preprocessing of Data<a id="section_PD"></a>

# ### Converting pdfs to image to text

# * using library   "from pdf2image- import convert_from_path" (pdf to image)
# * using pytesseract (image to text)
# * converted it to lowercase for the sake of analysis
# 
# #### [Index](#index)

# ### Generating all the text from the insurance pdfs and concatenating them

# In[4]:


get_ipython().run_cell_magic('time', '', 'tracemalloc.start()\n# Converting PDF\'s to text\nd_path = \'C:\\\\Users\\\\DharmendraGa_5wskc\\\\Downloads\\\\pdf_data\\\\Insurance\'\ndef image_2_text(pdf_path):\n    images = convert_from_path(pdf_path)\n    img = \'\\\\images\\\\page1.jpg\'\n    images[0].save(d_path+img,\'JPEG\')\n    image = d_path+img\n    text = pytesseract.image_to_string(Image.open(image),lang = "eng")\n    text = text.lower()\n    return text\n\ndef pdf_2_text(d_path):\n    text = ""\n    for (root,dirs,file) in os.walk(d_path):\n        for f in file:\n            if ".pdf" in f:\n                text += f"\\n{image_2_text(f)}"\n    return text\ntext = pdf_2_text(d_path)  \ntext')


# #### [Index](#index)

# ## Preprocessing using Regular Expressions <a id="section_NLP"></a>
# 
# ### 1.Tokenizing the Text (Splitting into Words)

# In[5]:


word_tokens = word_tokenize(text)
word_tokens


# In[6]:


# Extracting only text 
# filter_text = " ".join(re.findall("[a-zA-Z]+", text)).lower()
words = [w for w in word_tokens if len(w)>1 and len(w)<15]
words


# ### 2.Removing Stopwords

# In[7]:


# Stopwords in english
stop_words = stopwords.words('english')
print(stop_words)


# In[8]:


# remove stop words
words = [w for w in words if w not in stop_words]
words


# In[9]:


# Total filtered words
len(words)


# #### [Index](#index)

# ## Feature extraction<a id="section_FE"></a>

# ### Word Cloud with insurance pdfs (to know the best features of insurance data)<a id="p2"></a>

# In[10]:


# Wordcloud
@wordcloud
def display():
    print("Visualization!")
display()


# In[11]:


# Most Frequent words 
word_freq = Counter(words)
common_words = word_freq.most_common(10)
common_words


# In[12]:


# Selected features
features = [x[0] for x in common_words]
print(features)


# #### [Index](#index)

# ## Creating Dataset<a id="section_CD"></a>

# In[13]:


get_ipython().run_cell_magic('time', '', '# Extracting above features and creating dataframe\ntracemalloc.start()\nstart_FE = time.time()\n\n# Getting the pdf filenames \nos.chdir(\'C:\\\\Users\\\\DharmendraGa_5wskc\\\\Downloads\\\\pdf_data\\\\data\')\nfor _,_,f in os.walk(os.getcwd()):\n            pdf_files = f\n\npdf_files = [f for f in os.listdir()]\n\ndef extract_features(data):\n    # Empty Dataframe\n    df = pd.DataFrame()\n    # Creating sha256 object\n    sha256 = hashlib.sha256()\n    # Extracting Data\n    for i in range(len(data)):\n        text = image_2_text(data[i])\n        # Extracting Characters only \n        text = " ".join(re.findall("[a-zA-Z]+", text))\n        words = [w for w in text.lower().split() if len(w)>1]\n        row = {x : None for x in features}\n        for f in features:\n            if f in words:\n                row[f] = 1\n            else:\n                row[f] = 0\n        row[\'filename\'] = data[i]\n        # Ecrypting with SHA256\n        with open(data[i], \'rb\') as opened_file:\n            for line in opened_file:\n                sha256.update(line)\n            encrypted = sha256.hexdigest()\n        row[\'sha256\'] = encrypted\n        df = df.append(row, ignore_index=True)\n    num_cols = df.select_dtypes(include=np.number).columns.tolist()\n    df[num_cols] = df[num_cols].astype(int)\n    return df\n\n# Getting the dataframe           \ndf = extract_features(pdf_files)\nend_FE= time.time()\ntracemalloc.stop()')


# In[14]:


# Coloring DataFrame 
def color_df(val):
    if val == 0:
        color = 'blue'
    elif val==1:
        color = 'red'
    else:
        color = 'green'
    return 'color: %s' %color

df.style.applymap(color_df)


# In[15]:


# Memory Consumption for Feature Extraction
tracemalloc.stop()
mem_FE_c ,mem_FE_p = tracemalloc.get_traced_memory()
print("Current Memory used by Data Extraction and Engineering part ",mem_FE_c)
print("Peak Memory used by Data Extraction part during Execution",mem_FE_p)
# Time Taken for Feature Extraction
time_FE = end_FE-start_FE
print(f"Runtime of the Feature Extraction and Engineering Part is : {time_FE} secs")


# - The output is given in form of (current, peak),i.e, current memory is the memory the code is currently using and peak memory is the maximum space the program used while executing.
# 
# #### [Index](#index)

# ## Modelling 

# In[16]:


# Data 
df_raw = df.copy()
df_raw.head()


# In[17]:


# Filtering the dataframe
for x in df_raw.columns:
    y = str(df_raw[x][0])
    if re.search(".pdf",y):
        df = df_raw.drop([x],axis=1)
    elif bool(re.match("[a-z]", y)):
        z = x
        df.drop([x],axis=1, inplace=True)
    else:
        pass


# In[18]:


# To Transform into Numerical data
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('distilbert-base-nli-mean-tokens')
model.encode(df_raw[z])


# In[19]:


# Trying transformation for sha256
df[z] = df_raw[z].apply(lambda x: np.average(model.encode(x)))
df.head()


# #### [Index](#index)

# ## Correlation Heatmap<a id="p3"></a>

# In[20]:


# Heatmap
@heatmap
def display():
    print("Visualization!")
display()


# #### [Index](#index)

# ### Feature Selection<a id="section_FSE"></a>

# In[21]:


def feature_selection(data, pos_corr=0.6, neg_corr=-0.5):
    data = data[data.describe().columns.tolist()]
    # Create correlation matrix
    corr_matrix = data.corr()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find index of feature columns with correlation greater than 0.95
    selected_cols = [column for column in upper.columns if any(upper[column] > pos_corr) or any(upper[column] < neg_corr)]
    return selected_cols

# feature_selection(df, pos_corr=0.5, neg_corr=-0.1)
selected_feats = feature_selection(df)
selected_feats


# #### [Index](#index)

# ### Scaling<a id="section_FSC"></a>

# In[22]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler
num_cols = df.describe().columns.tolist()
def apply_scaling(data, scaler, feat_selection=None):
    if feat_selection == True:
        selected_feats = feature_selection(data)
        df = data[selected_feats]
    else:
        df = data[num_cols]
        
    s = scaler()
    s_data = s.fit_transform(df)
    scaled_df = pd.DataFrame(s_data, columns = df.columns)
    return scaled_df

# apply_scaling(df, MinMaxScaler, feat_selection=False)
apply_scaling(df, StandardScaler, feat_selection=True)


# #### [Index](#index)

# ## Clustering Techniques<a id="section_CT"></a>

# In[23]:


# Clustering Part 
start_cluster= time.time()
tracemalloc.start()


# In[24]:


from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN,AgglomerativeClustering


# In[25]:


import warnings 
warnings.filterwarnings('ignore')
# Cluster Modelling Function
def apply_clustering(algorithm=None,n_clusters=2,data=None,scale=None, feat_selection=None):
    # Applying Feature Selection
    if feat_selection == True:
        print("--"*11+"\nWith Feature Selection\n"+"--"*11)
        selected_feats = feature_selection(data)
        data = data[selected_feats]
    else:
        print("--"*13+"\nWithout Feature Selection\n"+"--"*13)
        data = data[data.describe().columns.tolist()]
        
    # Applying Scaling
    if scale == StandardScaler:
        data = apply_scaling(data, scale)
    elif scale == MinMaxScaler:
        data = apply_scaling(data, scale)
        
    # Applying Models
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
        print(f"\nN-Features: {model.n_features_in_}")
        print(f"\nN-Clusters: {n_clusters}")
        print(f"\nSilhouette coefficient: {silhouette_score(data,labels):0.2f}")
        print(f"\nInertia: {model.inertia_}")
        print(f"\n\nLabels : {labels}")
        return metrics
    elif algorithm in [DBSCAN, AgglomerativeClustering]:
        if algorithm == DBSCAN:
            model = algorithm(eps = 0.0375, min_samples = 5)
            model.fit(data)
            metrics = {"Model": model, "Silhouette Score":silhouette_score(data, model.labels_)}
        elif algorithm == AgglomerativeClustering:
            model = algorithm(affinity='euclidean', linkage='ward')  
            model.fit_predict(data)
            metrics = {"Model": model, "Silhouette Score":silhouette_score(data, model.labels_)}
        print(f"Model: {model}")
        print(f"\nN-Clusters: {n_clusters}")
        print(f"\nSilhouette coefficient: {silhouette_score(data, model.labels_)}")
        print(f"\nLabels: {model.labels_}")
        return metrics
apply_clustering(algorithm=KMeans, n_clusters=2, data=df, feat_selection=True)


# #### [Index](#index)

# ## Cluster Selection<a id="section_CS"></a>

# In[26]:


# Kmeans
import warnings
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist

distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 6)
num_cols = [c for c in df.describe().columns.tolist()]
for k in K:
    # Building and fitting the model
    kmeanModel = KMeans(n_clusters=k).fit(df[num_cols])
    kmeanModel.fit(df[num_cols])
 
    distortions.append(sum(np.min(cdist(df[num_cols], kmeanModel.cluster_centers_,
                                        'euclidean'), axis=1)) / df[num_cols].shape[0])
    inertias.append(kmeanModel.inertia_)
 
    mapping1[k] = sum(np.min(cdist(df[num_cols], kmeanModel.cluster_centers_,
                                   'euclidean'), axis=1)) / df[num_cols].shape[0]
    mapping2[k] = kmeanModel.inertia_
# Distortion based on k-values
for key, val in mapping1.items():
    print(f'Clusters : {key} :: distortion: {val}')


# **Distortion :**
# 
# It is calculated as the average of the squared distances from the cluster centers of the respective clusters
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

# #### [Index](#index)

# ### Elbow Method : To Select the K Value<a id="p4"></a>

# In[27]:


# Elbow Plot for above distortions w.r.t k-values
@elbowplot
def display():
    print("Visualization!")

display()


# ### Kneelocator : To locate knee<a id="p5"></a>

# In[28]:


# To Locate the Knee
from kneed import KneeLocator
@elbowKneeLocator
def display():
    print("Visualization!")

display()


# #### [Index](#index)

# ### Dendogram<a id="p6"></a>

# In[29]:


# Dendogram to select clusters
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))
plt.title("Dendrogram")
dend = shc.dendrogram(shc.linkage(df[num_cols], method='ward'),orientation="top")


# In[30]:


# Dendogram cluster cut 
plt.figure(figsize=(10, 7))  
plt.title("Dendrogram")  
dend = shc.dendrogram(shc.linkage(df[num_cols], method='ward'))
plt.axhline(y=6, color='r', linestyle='--');


# #### [Index](#index)

# ### Scatterplot<a id="p7"></a>

# In[31]:


# Scatterplot 
@scatterplot
def display():
    print("Visualization!")

display()


# In[32]:


# K-Means With all the features
apply_clustering(algorithm=KMeans, n_clusters=2, data=df, feat_selection=False)


# In[33]:


# K-Means With selective features
apply_clustering(algorithm=KMeans, n_clusters=2, data=df, feat_selection=True)


# ## Mini-Batch K-Means<a id="section_MBKM"></a>

# In[34]:


# MiniBatchKMeans With all the features
apply_clustering(algorithm=MiniBatchKMeans, n_clusters=2, data=df, feat_selection=False)


# In[35]:


# MiniBatchKMeans With selective features
apply_clustering(algorithm=MiniBatchKMeans, n_clusters=2, data=df, feat_selection=True)


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

# #### [Index](#index)

# ## DBSCAN <a id="section_DB"></a>

# In[36]:


# DBSCAN With selective features
apply_clustering(algorithm=DBSCAN, n_clusters=2, data=df, feat_selection=True)


# ## Agglomerative Clustering<a id="section_A"></a>

# In[37]:


# Agglomerative With selective features
apply_clustering(algorithm=AgglomerativeClustering, n_clusters=2, data=df, feat_selection=True)


# #### [Index](#index)

# ### Silhoutte Visualization<a id="p8"></a>

# In[38]:


# Calling Silhouette Score Plot 
@silhouetteplot
def display():
    print("Visualization!")
display()


# #### [Index](#index)

# ## Best Model<a id="BM"></a>

# In[39]:


# Finding the Best Model 
models = [KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering]
metrics_df = pd.DataFrame()
for m in models: 
    print("--"*50)
    metrics = apply_clustering(algorithm=m, n_clusters=2, data=df, feat_selection=True)
    metrics_df = metrics_df.append(metrics, ignore_index=True)  
# Finding the Best Model 
models = [KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering]
metrics_df2 = pd.DataFrame()
for m in models: 
    print("--"*50)
    metrics = apply_clustering(algorithm=m, n_clusters=2, data=df, feat_selection=False)
    metrics_df2 = metrics_df2.append(metrics, ignore_index=True)  
metrics_df2['Silhouette After Feature Engineering'] = metrics_df['Silhouette Score']
metrics_df2.rename(columns = {'Silhouette Score':'Silhouette Before Feature Engineering'}, inplace=True)
metrics_df2


# In[40]:


# Silhouette Score Plot
labels = ['K-Means', 'MiniBatchK-Means', 'DBSCAN', 'Agglomerative']
plt.title('Silhouette Score for Clustering Models')
plt.plot(range(1,5), metrics_df2['Silhouette Before Feature Engineering'] , label = metrics_df2.columns.tolist()[1])
plt.plot(range(1,5), metrics_df2['Silhouette After Feature Engineering'], label = metrics_df2.columns.tolist()[2])
plt.legend()
plt.xticks(range(1,5), labels, rotation=20)
plt.ylabel('Silhouette')
plt.margins(0.2)
plt.subplots_adjust(bottom = 0.15)
plt.show()


# **Time & Space checkpoint**

# In[41]:


end_cluster = time.time()
time_CT = end_cluster-start_cluster
print(f"Runtime of the Cluster Modelling Part is : {time_CT} secs")

mem_CT_c ,mem_CT_p= tracemalloc.get_traced_memory()
print("\nCurrent memory used by Cluster modelling part: ",mem_CT_c)
print("Peak memory used by Cluster modelling part during Execution: ",mem_CT_p)
tracemalloc.stop()


# - The output is given in form of (current, peak),i.e, current memory is the memory the code is currently using and peak memory is the maximum space the program used while executing.

# #### [Index](#index)

# ## Classification Algorithms<a id="section_CLA"></a>

# ### Importing dataset using target variables(added manually)

# In[42]:


# Starting 
start_class= time.time()
tracemalloc.start()


# In[43]:


# Creating Data
data = df.copy()
for m in metrics_df[metrics_df['Silhouette Score']==metrics_df['Silhouette Score'].max()]['Model']:
    m.fit(data)
    labels = m.labels_
# Creating Target by best clustering model labels 
data['Target'] = pd.Series(abs(labels))
data


# In[44]:


data.info()


# In[45]:


# Value Counts
data['Target'].value_counts()


# #### [Index](#index)

# ### Countplot<a id="p9"></a>

# In[46]:


# Countplot for target
sns.countplot(data['Target'])
plt.title('Insurance vs Non-Insurance');


# In[47]:


data.describe()


# #### [Index](#index)

# ## Splitting of data into train and test

# In[48]:


from sklearn.model_selection import train_test_split
X=data.iloc[:,:-1]
y=data["Target"]
X_train, X_test, y_train, y_test = train_test_split( X,y , test_size = 0.25, random_state = 5) 
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)


# In[49]:


# Scaling the data
from sklearn.preprocessing import MinMaxScaler# set up the scaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# #### [Index](#index)

# ### KNN<a id="section_KNN"></a>

# In[50]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)


# In[51]:


print(knn.score(X_test, y_test))


# ### KNN Performance plot<a id="p10"></a>

# In[52]:


# find k value
score_list = []
for each in range(1,15):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(X_train,y_train)
    score_list.append(knn2.score(X_test,y_test))
    
plt.plot(range(1,15),score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show();


# #### [Index](#index)

# ### Logistic Regression<a id="section_LR"></a>

# In[53]:


from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
y_logistic_pred = model.predict(X_test)
logistic_re = classification_report(y_test,y_logistic_pred)
print(logistic_re)


# ### Support Vector Machine<a id="section_SVM"></a>

# In[54]:


# support vector classifier
from sklearn.svm import SVC 
model = SVC()
model.fit(X_train,y_train)
predictions = model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix,plot_confusion_matrix
print(classification_report(y_test,predictions))


# #### [Index](#index)

# ### SVM Performance<a id="p11"></a>

# In[55]:


# Confusion matrix
@confusionMatrix
def display():
    print("Visualization!")

display()


# ### Decision Tree<a id="section_DT"></a>

# In[56]:


# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
treeClassifier = DecisionTreeClassifier(max_depth=3, min_samples_leaf=6)
treeClassifier.fit(X_train,y_train)
# prediction 
y_pred = treeClassifier.predict(X_test)
# Check classification report with confusion matrix and report
from sklearn.metrics import classification_report,confusion_matrix
report = classification_report(y_test,y_pred)
print(report)


# #### [Index](#index)

# ### Decision Tree Plot<a id="p12"></a>

# In[57]:


# Tree Plot
from sklearn.tree import plot_tree
@decisionTree
def display():
    print("Visualization!")

display()


# ### Random Forest<a id="section_RF"></a>

# In[58]:


# Random Forest 
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=40)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
cr = classification_report(y_test,rfc_pred)
print(cr)


# #### [Index](#index)

# ### Random Forest Plot<a id="p13"></a>

# In[59]:


@randomForestTree
def display():
    print("Visualization!")

display()


# In[60]:


# Time
end_class=time.time()
time_CM = end_class-start_class
print(f"Runtime of classification part is : {time_CM} secs")
# Memory
mem_CM_c ,mem_CM_p= tracemalloc.get_traced_memory()
print("\nCurrent memory used by Classification modelling part ",mem_CM_c)
print("Peak memory used by Classification modelling part during Execution",mem_CM_p)
tracemalloc.stop()


# #### The output is given in form of (current, peak),i.e, current memory is the memory the code is currently using and peak memory is the maximum space the program used while executing.

# #### [Index](#index)

# In[61]:


# Time and Space Calculations
def time_memory():
    a = {"currentMemory":[mem_FE_c,mem_CT_c,mem_CM_c],
    "peakMemory":[mem_FE_p,mem_CT_p,mem_CM_p],
    "timeTaken":[time_FE,time_CT,time_CM]}
    index_ = ["Data Engineering","Clustering","Classification"]
    tm = pd.DataFrame(a,index = index_)
    print(tm)
    
time_memory()


# -  A kibibit (a contraction of kilo binary digit) or sometimes Kib.
# -  1 Kib = 128 bytes.
# -  1 MB  = 7812.5 Kib

# ### Learning Outcomes : <a id="section_LO"></a>
#   While feature extraction , we learned about :
# * sha256 & hashlib library
# * pdf2image library (how a pdf document can be converted to image)
# * Image processing & text recognition using OCR library Tesseract of Google and Python Imaging Library(PIL)
# * Different Natural Language Processing (NLP) techniques for cleaning and extracting features from the content of the document.
# * Kmeans clustering
# * Elbow method for finding number of clusters
# * Dendograms for visualizing the hierarchical clustering
# * Agglomerative Clustering
# * Mini Batch Kmeans
# * Silhoutte metrics & Visualization
# * DBScan
# * Classification algorithms for testing the clusters
# * Time Complexity
# * Tracemalloc,that traces every memory block in python

# #### [Index](#index)

# ## References<a id="section_REF"></a>

# * Google images
# * Stack Overflow for errors
# * https://www.lucidchart.com - for flow diagram
# * https://github.com/fabiomus - for sha256
# * https://www.youtube.com/watch?v=SVmZZK11Tjc - for sha256
# * https://www.geeksforgeeks.org/ 
# * https://scikit-learn.org/stable/index.html
# * https://github.com/utkuozbulak/unsupervised-learning-document-clustering

# ### Observations in Jupyter Notebook:
#     
# **1.Package Installation:**
# - !pip install plotly 
# -  pip install re
# 
#    ! -- To run the command as bash/shell we use this '!' exclamation mark before pip, without that it will be just notebook command 
# 
# **2.Merging Celss:**
# 
# - press shift and select cells and press 'M' to Merge cells
# 
# **3.Hyper Linking Between Cells**

# #### [Index](#index)

# In[ ]:





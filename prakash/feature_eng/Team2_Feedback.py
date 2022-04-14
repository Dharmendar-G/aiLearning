#!/usr/bin/env python
# coding: utf-8

# TEAM-2 :
# 
#  prakash.g@bourntec.in ,shaikmahaboob.b@bourntec.in, shilpa.s@bourntec.in

# #Problem Statement
# 
# 
# 
# Feature Engineering Techniques:
# 
# ->>> Multi Variate Analysis
# 
# ->>> Bi_variate Analysis b/w the Predictor variables and blw the predictors variables and target columnq
# 
# ->>> finding Relationship and degree of Relation
# 
# ->>> if Any Leverage points Visulize the Analysis Box plots and pair plots,histogram,density curves
# 
# Here we have collected the Market Analysis
# 
# it is all in the data set: Marketing_Analysis.csv
# 
# Steps to follows:
# 
# 'Data Source'
# 
# 'Data Cleaning'
# 
# 'Univariate_Analysis'
# 
# 'Bi_variate Analyis'
# 
# 'Multi_variate Analyis'

# # Let's get started!
# 
# Check out the data
# We've been able to get some data Market Analysis as a csv set, let's get our environment ready with the libraries we'll need and then import the data!

# # Data Analysis Phase

# In[1]:


# Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Read the data set of "Marketing Analysis" in data.
data= pd.read_csv("Marketing_Analysis.csv",skiprows = 2)

#printing top 5 records 
data.head()


# In[3]:


#checking the no.of rows and columns present in the dataset
data.shape


# # EDA Part

# In[4]:


#now check the type of attributes in the given dataset
data.info()


# In[5]:


#finding missing values
data.isnull().sum()


# In[6]:


#statistical analysis
data.describe()


# In[7]:


#no.of unique values
data.apply(lambda x: len(x.unique()))


# In[8]:


#value_counts of each column
for col in data.columns:
    #print(col)
    print(data[col].value_counts())
    


# In[9]:


#making the list of features which has missing values
feat_with_na = [feature for feature in data.columns if data[feature].isnull().sum()>1]
feat_with_na


# In[10]:


#percentage of missing values
for feature in feat_with_na:
    print(feature,np.round(data[feature].isnull().mean(),4),'% missing values')


# In[11]:


#list of all numerical_columns
num_feat =[feature for feature in data.columns if data[feature].dtypes!='object']
print('no.of numerical_features:',len(num_feat))
print('numerical_features are',num_feat)


# In[12]:


#temporal variables [ex: Datetime variables]
#list of variables that contain year information
year_feat = [feature for feature in  data.columns if 'year' in feature or 'month' in feature]
year_feat


# In[13]:


#list of discrete features
discrete_feat = [feature for feature in num_feat if len(data[feature].unique())<25 and feature not in year_feat+['id']]
discrete_feat


# In[14]:


#list of continuous variables
conti_feat =[feature for feature in num_feat if feature not in discrete_feat +year_feat+['id']]
print('no.of conti_features:',len(conti_feat))
print('conti_features are',conti_feat)


# In[15]:


#list of categorical variables
cat_feat =[feature for feature in data.columns if data[feature].dtypes == 'object']
print('no.of categorical_features:',len(cat_feat))
print('categorical_features are',cat_feat)


# In[16]:


#cardinality
for feature in cat_feat:
    print('the feature is {} and no.of categories are {}'.format(feature,len(data[feature].unique())))


# In[17]:


#note : if the no.of categories are <8 means we go with OHE 


# In[18]:


for feature in cat_feat:
    if len(data[feature].unique()) <8:
           print(feature)
    #print('the feature is {} and no.of categories are {}'.format(feature,len(data[feature].unique())))


# In[19]:


#outliers
for feature in num_feat:
    print('Q1_'+feature ,data[feature].quantile(0.25))
    print('Q3_'+feature ,data[feature].quantile(0.75))
    q1=data[feature].quantile(0.25)
    q3=data[feature].quantile(0.75)
    IQR=q3-q1
    print('IQR_'+feature,IQR)
    lower_limit = q1 - (1.5*IQR)
    upper_limit = q3 + (1.5*IQR)
    print('Low_limit_'+feature,lower_limit)
    print('up_limit_'+feature,upper_limit)
    out_upper=len(data[feature][data[feature] > upper_limit])
    out_lower=len(data[feature][data[feature] < lower_limit])
    print('the outliers of upper limit is:',out_upper)
    print('the outliers of lower limit is:',out_lower)
    total_outliers = out_upper+out_lower
    print('The outlayers present in_ '+feature,total_outliers)
    #for feature in feat_with_na:
    print(feature,'column contains',np.round((total_outliers/len(data[feature])*100),4),'% outliers')
    print()


# In[20]:


#outliers
#import numpy as np
for feature in conti_feat:
    df = data.copy()
    if 0 in df[feature].unique():
        pass
    else:
        df[feature]=np.log(df[[feature]])
        df.boxplot(column =feature)
        print('Q1_'+feature ,df[feature].quantile(0.25))
        print('Q3_'+feature ,df[feature].quantile(0.75))
        print('The outlayers present in_ '+feature,total_outliers)
        print(feature,'column contains',np.round((total_outliers/len(data[feature])*100),4),'% outliers')
        plt.show()
        
    


# In[21]:


#create response_rate of numerical data type where response "yes"= 1, "no"= 0
data['response_rate'] = np.where(data.response=='yes',1,0)
data.response_rate.value_counts()


# # Visualizations

# #univarient analysis

# In[22]:


#using decorators we are plotting all categorical features


# In[23]:


for feature in cat_feat:
    def bar_plot(func):
        def barplot():
            plt.figure(figsize=(15,6))
            plt.subplot(1,2,1)
            print(data[feature].value_counts(normalize=True).plot.bar())
            #func()
            print("barplot below")
        return barplot()
    
    def pie_plot(func):
        def pieplot():
            plt.figure(figsize=(15,6))
            plt.subplot(1,2,1)
            print(data[feature].value_counts(normalize=True).plot.pie())
            #func()
            print("pieplot below")
        return pieplot()
    
    def count_plot(func):
        def countplot():
            plt.figure(figsize=(15,6))
            plt.subplot(1,2,1)
            print(sns.countplot(x=data[feature],data=data))
            func()
            print("countplot below")
        return countplot()
    
    @bar_plot
    @pie_plot
    @count_plot
    def plotting():
        print("ALL_PLOTS_BELOW")  


# In[24]:


#using decorators we are plotting all numerical features


# In[25]:


for feature in num_feat:
    def hist_plot(func):
        def histplot():
            print(data[feature].hist(bins=25))
            plt.xlabel(feature)
            plt.ylabel('count')
            plt.title(feature)
            plt.show()
        return histplot() 
    
    def dist_plot(func):
        def distplot():
            print(sns.FacetGrid(data,hue="response",size=5).map(sns.distplot,feature).add_legend())
            plt.xlabel(feature)
            plt.ylabel('count')
            plt.title(feature)
            plt.show()
        return distplot()
    
    def box_plot(func):
        def boxplot():
            print(sns.boxplot(x=feature,data=data))
            plt.xlabel(feature)
            plt.ylabel('count')
            plt.title(feature)
            plt.show()
        return boxplot()
    
    def violin_plot(func):
        def violinplot():
            print(sns.violinplot(x=feature,data=data))
            plt.xlabel(feature)
            plt.ylabel('count')
            plt.title(feature)
            plt.show()
        return violinplot()
    
    def scatter_plot(func):
        def scatterplot():
            print(sns.scatterplot(data[feature],y=data['response'],data=data))
            plt.xlabel(feature)
            plt.ylabel('count')
            plt.title(feature)
            plt.show()
        return scatterplot()
    
    def lm_plot(func):
        def lmplot():
            print(sns.lmplot(x =feature, y ='response_rate', data = data))
            plt.xlabel(feature)
            plt.ylabel('count') 
            plt.title(feature)
            plt.show()
        return lmplot()
    
    
            
    @hist_plot
    @dist_plot
    @box_plot
    @violin_plot
    @scatter_plot
    @lm_plot
    def plotting():
        print("ALL_PLOTS_BELOW") 


# In[ ]:





# In[26]:


#countplot for categorical features
for feature in cat_feat:
    sns.countplot(x=data[feature],data=data)
    plt.xlabel(feature)
    plt.ylabel('count')
    plt.title(feature)
    plt.show()


# In[27]:


#barplot for categorical features
for feature in cat_feat:
    data[feature].value_counts(normalize=True).plot.bar()
    plt.show()


# In[28]:


#pie chart for categorical features
for feature in cat_feat:
    data[feature].value_counts(normalize=True).plot.pie(autopct='%.2f')
    plt.show()


# In[29]:


#histogram for numerical features
for feature in num_feat:
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel('count')
    plt.title(feature)
    plt.show()
    


# In[30]:


#distribution plot for numerical features
for feature in num_feat:
    sns.FacetGrid(data,hue="response",size=5).map(sns.distplot,feature).add_legend()


# In[31]:


#box plot for univarient analysis
for feature in num_feat:
    sns.boxplot(x=feature,data=data)
    #plt.xlabel(feature)
    plt.title(feature)
    plt.show()


# In[32]:


#violin plot for univarient analysis
for feature in num_feat:
    sns.violinplot(x=feature,data=data)
    #plt.xlabel(feature)
    plt.title(feature)
    plt.show()


# #bivarient analysis

# In[33]:


#scatter plot for bivarient analysis
for feature in num_feat:
    sns.scatterplot(data[feature],y=data['response'],data=data)
    plt.xlabel(feature)
    #plt.ylabel('count')
    plt.title(feature)
    plt.show()


# In[34]:


#boxplot for bivarient analysis
for feature in num_feat:
    sns.boxplot(x='response',y=feature,data=data)
    plt.xlabel(feature)
    plt.ylabel('response')
    plt.title(feature)
    plt.show()


# In[35]:


#violinplot for bivarient analysis
for feature in num_feat:
    sns.violinplot(x='response',y=feature,data=data)
    plt.xlabel(feature)
    plt.ylabel('response')
    plt.title(feature)
    plt.show()


# In[36]:


#bivarient analysis of lm plot
for feature in num_feat:
    sns.lmplot(x =feature, y ='response_rate', data = data)
    plt.show()


# In[37]:


#visualizing the pair plot for every column with hue of response
sns.pairplot(data, hue="response",height=3)
plt.show()


# # Feature Engineering

# In[38]:


#splitting the combined columns
#Extract job  & Education in newly from "jobedu" column.
data['job']= data["jobedu"].apply(lambda x: x.split(",")[0])
data['education']= data["jobedu"].apply(lambda x: x.split(",")[1])


# In[39]:


#droping unwanted columns 
#Drop the customer id as it is of no use.
data.drop('customerid', axis = 1, inplace = True)


# #handling missing values:
# using fillna() or
# using dropna() or
# using mean,median,mode

# In[40]:


#droping the missing values
data = data.dropna()


# In[41]:


#rechecking the null values
data.isnull().sum()


# In[ ]:





# #handling outliers
# using percentile method
# using standard deviation method
# using winsorizing method etc

# In[42]:


#using winsorizing technique(for upper limit outliers)
import scipy.stats
for feature in conti_feat:
    if feature !='customerid':
        print(scipy.stats.mstats.winsorize(data[feature],limits=0.95))


# In[43]:


#printing the last 5 records
data.tail()


# In[44]:


data.shape


# #label encoding

# In[50]:


#label encoding
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
le = LabelEncoder()
for feature in cat_feat:
    data[feature]=le.fit_transform(data[feature].astype(int))


# In[51]:


#printing the top 5 records
data.head()


# In[52]:


corr = data.corr()
plt.figure(figsize=(16,8))
sns.heatmap(corr,annot=True,cmap='coolwarm')


# #Feature selection

# In[53]:


#selecting features using correlation matrix
def correlation(data,threshold):
    col_corr= set()
    corr_mat = data.corr()
    for i in range(len(corr_mat.columns)):
        for j in range(i):
            if abs(corr_mat.iloc[i,j]) > threshold:
                colname = corr_mat.columns[i]
                col_corr.add(colname)
    return  col_corr         


# In[54]:


#calling function passing parameters 
corr_feat = correlation(data,0.5)


# In[55]:


corr_feat


# In[56]:


#statistical analysis of data
def get_summary_stastics(data):
    mean = np.round(np.mean(data),2)
    #median =np.round(np.median(data),2)
    #min_val = np.round(np.min(data),2)
    #max_val =np.round(np.max(data),2)
    Q1 = np.round(data.quantile(0.25),2)
    Q3 = np.round(data.quantile(0.75),2)
    #IQR
    iqr = np.round(Q3 - Q1,2)
    #print('min: %s', min_val)
    #print('max: %s', max_val)
    print('mean: %s', mean)
    print('25th per: %s',Q1)
    print('75th per: %s', Q3)
    print('IQR: %s',iqr)


# In[57]:


get_summary_stastics(data)


# In[ ]:





# # feedbacks : 

# In[59]:


# 1)how much memory is used for this notebook?
# 2)how much time does it takes to execute this notebook?
# 3)garbage collector after executing the notebook?
# 4)list out the categorical and numerical visulization and make a decorator?
#Pending
# 5)how can we run 2 parallel different notebooks(categorical and numerical)


# In[61]:


#1
pip install memory_profiler


# In[68]:


import memory_profiler
from memory_profiler import profile

@profile
def main_func():
    import random
    arr1 = [random.randint(1,10) for i in range(100000)]
    arr2 = [random.randint(1,10) for i in range(100000)]
    arr3 = [arr1[i]+arr2[i] for i in range(100000)]
    del arr1
    del arr2
    tot = sum(arr3)
    del arr3
    print("The total size of the file in KB",tot)

# if _name_ == "_main_":
main_func()


# In[69]:


import sys
# These are the usual ipython objects, including this one you are creating
ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']

# Get a sorted list of the objects and their sizes
sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)


# In[60]:


# 2.import timeit

def test(n):
    return sum(range(n))

n = 10000
loop = 1000

result = timeit.timeit('test(n)', globals=globals(), number=loop)
print(result / loop)


# In[6]:


#3.Garbage collection
#Import necessary libraries

import pandas as pd
import sys  #system specific parameters and names
import gc   #garbage collector interface
file_path='Marketing_Analysis.csv'
df=pd.read_csv(file_path, low_memory=False)
df.head(3)
memory_usage_by_variable=pd.DataFrame({k:sys.getsizeof(v) for (k,v) in locals().items()},index=['Size'])
memory_usage_by_variable=memory_usage_by_variable.T
memory_usage_by_variable=memory_usage_by_variable.sort_values(by='Size',ascending=False).head(10)
memory_usage_by_variable.head()


# In[7]:


# import gc
# gc.collect()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:
#the end





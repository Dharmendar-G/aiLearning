#importing required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
#reading the dataset
data = pd.read_csv('Bank_Personal_Loan_Modelling.csv')
data.head()
print(data.head())
#checking no.of rows and columns
data.shape
#list of data columns
list(data.columns)
#checking info
data.info()
#statistical summary  of dataset
data.describe()
#checking null values
data.isnull().sum()
#no.of unique values
data.apply(lambda x: len(x.unique()))
# value counts in each column
for i in data.columns:
    print(data[i].value_counts())

#list of categorical columns
cat_feats = []
for x in data.dtypes.index:
    if data.dtypes[x] == 'object':
        cat_feats.append(x)

cat_columns = [x for x in data.columns if data[x].dtypes == 'O' ]
#list of all numerical_columns
num_columns = [x for x in data.columns if data.dtypes[x] !='object']
print(num_columns)
print(len(num_columns))
#or
num_feats = []
for col in data.dtypes.index:
    if data.dtypes[col] != 'object':
        num_feats.append(col)
#list of all numerical_columns
num_columns = [x for x in data.columns if data.dtypes[x] !='object']
print(num_columns)
print(len(num_columns))

#list of discrete features
discrete_feat = [feature for feature in num_feats if len(data[feature].unique())<25 ]
discrete_feat
#list of continuous variables
conti_feat =[feature for feature in num_feats if feature not in discrete_feat ]
print('no.of conti_features:',len(conti_feat))
print('conti_features are',conti_feat)
#outliers
for feature in num_feats:
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

#value counts of personal loan
data['Personal Loan'].value_counts()

#visualizations
#distribution plot
for feature in num_feats:
    sns.distplot(data[feature])
    plt.show()

#distribution plot using face grid
for feature in num_feats:
    sns.FacetGrid(data,hue="Personal Loan",size=5).map(sns.distplot,feature).add_legend()
#barplot
for feature in num_feats:
    plt.figure(figsize = (20,10))
    data[feature].value_counts(normalize=True).plot.bar()
    plt.xlabel(feature)
    plt.ylabel('count')
    plt.show()

#histogram
for feature in num_feats:
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel('count')
    plt.show()

# pieplot
for feature in num_feats:
    plt.figure(figsize=(25, 10))
    plt.subplot(1, 2, 1)
    print(data[feature].value_counts(normalize=True).plot.pie(autopct='%.2f'))
    plt.show()

#countplot
for feature in num_feats:
    plt.figure(figsize=(15,6))
    #plt.subplot(1,2,1)
    sns.countplot(x=data[feature],data=data)
    plt.show()
#boxplot
for feature in num_feats:
    sns.boxplot(x = feature,data=data)
    plt.xlabel(feature)
    plt.show()

#violinplot
for feature in num_feats:
    sns.violinplot(x = feature,data=data)
    plt.xlabel(feature)
    plt.show()

#scatterplot
for feature in num_feats:
    sns.scatterplot(x=feature,y='Personal Loan',data=data)
    plt.xlabel(feature)
    plt.ylabel('Personal Loan')
    plt.show()

#lmplot
for feature in num_feats:
    sns.lmplot(x =feature, y ='Personal Loan', data = data)
    plt.xlabel(feature)
    plt.ylabel('count')
    plt.title(feature)
    plt.show()
#pairplot
sns.pairplot(data,hue='Personal Loan',height = 5)
plt.show()

#handling missing values
data.isnull().sum()
#handling outliers using winsorizing technique
import scipy.stats
for feature in num_feats:
    print(scipy.stats.mstats.winsorize(data[feature],limits=0.95))
#dropping unwanted columns
data.drop(['ID'],axis =1,inplace =True)
#correlation
corr = data.corr()
corr

#heatmap
plt.figure(figsize=(20,10))
sns.heatmap(corr,annot = True)
plt.show()

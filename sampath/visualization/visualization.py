import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('visualization.csv')

df.head()

## TYPE

#check the unique elements
df.Type.value_counts()

#plots a barplot wrt TYPE column
plt.figure(figsize=(16,6))
vals=df['Type'].value_counts()
sns.barplot(vals,vals.index,palette='rocket')
plt.xlabel('Range')

#plots a pieplot wrt TYPE column with their respective percentages
plt.figure(figsize=(14,9))
df.Type.value_counts().plot(kind='pie',autopct='%.0f%%',fontsize=12)
plt.title('Types of Products', fontsize=25)

## VENDOR

##check the unique elements
df.Vendor.value_counts()


##plots a barplot wrt Vendor column
## Takes only top 20 Vendors in terms of occurence
plt.figure(figsize=(16,6))
vals=df['Vendor'].value_counts()[:20]
sns.barplot(vals,vals.index)
plt.xlabel('VENDORS')
plt.grid(True)

##plots a pieplot wrt VENDORS column with their respective percentages
#Taken only top 15 most occured VENDORS from the data
plt.figure(figsize=(18,10))
df.Vendor.value_counts()[:15].plot(kind='pie',autopct='%.0f%%',fontsize=12)
plt.title('Types of Vendros', fontsize=25)



## APPNAME

###check the unique elements and count of each unique element
df.appName.value_counts()

##plots a barplot wrt appName column
## Takes only top 15 APPS in terms of occurence
plt.figure(figsize=(16,6))
vals=df['appName'].value_counts()[:15]
sns.barplot(vals,vals.index,palette='cubehelix')
plt.xlabel('appName')
plt.grid(True)

##plots a pieplot wrt appName column with their respective percentages
#Taken only top 15 most occured APPS from the data
plt.figure(figsize=(10,10))
df.appName.value_counts()[:15].plot(kind='pie',autopct='%.0f%%',fontsize=12,)
plt.title('Types of APPS', fontsize=25)

## VERSION

##check the unique elements and count of each unique element
df.Version.value_counts()

##plots a barplot wrt Version column
## Takes only top 15 Versions in terms of occurence
plt.figure(figsize=(16,6))
vals=df['Version'].value_counts()[:15]
sns.barplot(vals,vals.index)
plt.xlabel('VERSION')
plt.grid(True)
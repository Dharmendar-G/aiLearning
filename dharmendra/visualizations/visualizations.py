## Visualizations for NVD Data

# Libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
plt.style.use('fivethirtyeight')

# Loading data
df = pd.read_csv('visualization.csv')
df = df.drop(['Unnamed: 0'], axis=1)
df

# information
df.info()

# description
df.describe()

# value counts 
for c in df.columns.tolist():
    print('--'*50,f'\nColumn :{c}\n')
    print(df[c].value_counts())

### Missing Data Visualization

# Missing values for each variable
total = df.isnull().sum().sort_values(ascending = False)
missing = (df.isna().sum()/df.shape[0]*100).sort_values(ascending=False)
m = pd.concat([total, missing], axis=1, keys=['Total', 'Missing'])
sns.set(style = 'darkgrid')
plt.figure(figsize = (8, 4))
plt.xticks(rotation='40')
sns.barplot(x=m.index, y=m["Missing"],color="g",alpha=0.8)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent', fontsize=15)
plt.title('Missing Data', fontsize=15);

## Univariate Analysis

### Top Vendors

# Barplot
v_count = df.groupby("Vendor")["appName"].count()
c = v_count[v_count>2000].sort_values().plot(kind='barh',figsize=(15, 10), color='#619CFF', zorder=2)
c.set_ylabel('Vendors')
plt.title('Top Vendors for Cyber Security/Other Firmwares', fontsize=15)
plt.show();

# Pie Chart
top_vendors = pd.DataFrame(v_count[v_count>5000]).index.tolist()
v_df = df[df['Vendor'].isin(top_vendors)]
plt.figure(figsize=(10,8))
v_df['Vendor'].value_counts()[::-1].plot(kind='pie', title=c, autopct="%.1f%%", fontsize=12)
plt.title('Top Percentage Vendors', fontsize=20)
plt.tight_layout()

# Countplot
top_vendors = pd.DataFrame(v_count[v_count>2000]).index.tolist()
v_df = df[df['Vendor'].isin(top_vendors)]
plt.figure(figsize=(20, 10))
sns.countplot(x='Vendor', data=v_df)
plt.xticks(rotation=90)
plt.title("Top Vendors", fontsize = 30)
plt.show();

### Top Applications 

# Barplot
a_count = df.groupby("appName")["Vendor"].count()
c = a_count[a_count>1000].sort_values().plot(kind='barh',figsize=(15, 10), color='#619CFF', zorder=2)
c.set_ylabel('Apps')
plt.title('Top Apps')
plt.show();

# Pie Chart
top_apps = pd.DataFrame(a_count[a_count>2000]).index.tolist()
a_df = df[df['appName'].isin(top_apps)]
plt.figure(figsize=(10,8))
a_df['appName'].value_counts()[::-1].plot(kind='pie', title=c, autopct="%.1f%%", fontsize=12)
plt.title('Top Percentage Apps', fontsize=20)
plt.tight_layout()

# Top Applications 
top_apps = pd.DataFrame(a_count[a_count>1000]).index.tolist()
a_df = df[df['appName'].isin(top_apps)]
plt.figure(figsize=(20, 10))
sns.countplot(x='appName', data=a_df)
plt.xticks(rotation=90)
plt.title("Top Applications", fontsize = 30)
plt.show();

### Type of Product

# Countplot for Type
plt.figure(figsize=(8, 6))
sns.countplot(x='Type', data=v_df)
plt.title("Category Type", fontsize = 25)
plt.show();

# Pichart for Type
plt.figure(figsize=(10,8))
df['Type'].value_counts()[::-1].plot(kind='pie', title=c, autopct="%.0f%%", fontsize=12)
plt.title('Different Type of Products', fontsize=20)
plt.tight_layout()

# Barplot
t_count = df.groupby("Type")["Vendor"].count()
t_count.sort_values().plot(kind='barh',figsize=(10, 6), color='#619CFF', zorder=2)
plt.title('Type of Product', fontsize=20)
plt.show();
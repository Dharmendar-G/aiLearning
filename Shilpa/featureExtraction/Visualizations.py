import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
df = pd.read_csv('visualization.csv')
print(df['CPEMatchString'])
print(df['CPEMatchString'][165689])
df = pd.read_csv('visualization.csv')
df = df.drop(['Unnamed: 0'], axis=1)
df
df.info()#information
df.describe()# description

# value counts
for c in df.columns.tolist():
    print('--'*50,f'\nColumn :{c}\n')
    print(df[c].value_counts())
    total = df.isnull().sum().sort_values(ascending = False)
missing = (df.isna().sum()/df.shape[0]*100).sort_values(ascending=False)
m = pd.concat([total, missing], axis=1, keys=['Total', 'Missing'])
sns.set(style = 'darkgrid')
plt.figure(figsize = (8, 4))
plt.xticks(rotation='30')
sns.barplot(x=m.index, y=m["Missing"],color="g",alpha=0.6)
plt.xlabel('Features', fontsize=8)
plt.ylabel('Percent', fontsize=8)
plt.title('Missing Data', fontsize=10);
surveys_df = pd.read_csv("visualization.csv")
surveys_df
surveys_df.head()
df.isnull().any()
df.head()
Type= df["Type"]
Version = df["Version"]
x = []
y = []
type(Type)
x=list(Type)
y=list(Version)
plt.bar(x,y,data=df)
plt.scatter(x,y)
plt.xlabel('CPEMatchString')
plt.ylabel('Version')
plt.title('Data')
plt.show()
Type= df["Type"]
Vendor = df["Vendor"]
x = []
y = []

x=list(Type)
y=list(Vendor)
plt.scatter(x,y)
plt.xlabel('Type')
plt.ylabel('Vendor')
plt.title('Data')
plt.show()
appName= df["appName"]
Version = df["Version"]
x = []
y = []
x=list(appName)
y=list(Version)
plt.scatter(x,y)
plt.xlabel('appName')
plt.ylabel('Version')
plt.title('Data')
plt.show()
# Count the number of samples by version
survey = surveys_df.groupby('CPEMatchString')['Version'].count()
print(Version)
# value counts
for c in df.columns.tolist():
    print('--'*50,f'\nColumn :{c}\n')
    print(df[c].value_counts())
    # Pichart for Type
    plt.figure(figsize=(8, 6))
    df['Type'].value_counts()[::-1].plot(kind='pie', title=c, autopct="%.0f%%", fontsize=12)
    plt.title('Different Type of Products', fontsize=20)
    plt.tight_layout()
    # Barplot
    t_count = df.groupby("Type")["Vendor"].count()
    t_count.sort_values().plot(kind='barh', figsize=(8, 4), color='#619CFF', zorder=2)
    plt.title('Type of Product', fontsize=20)
    plt.show();
    a_count = df.groupby("appName")["Vendor"].count()
    c = a_count[a_count > 1000].sort_values().plot(kind='barh', figsize=(15, 10), color='#619CFF', zorder=2)
    c.set_ylabel('Apps')
    plt.title('Top Apps')
    plt.show();
    t_count = df.groupby("Type")["Version"].count()
    t_count.sort_values().plot(kind='barh', figsize=(10, 6), color='#619CFF', zorder=2)
    plt.title('Type of App', fontsize=20)
    plt.show();
    # Pie Chart
    top_apps = pd.DataFrame(a_count[a_count > 2000]).index.tolist()
    a_df = df[df['appName'].isin(top_apps)]
    plt.figure(figsize=(6, 4))
    a_df['appName'].value_counts()[::-1].plot(kind='pie', title=c, autopct="%.1f%%", fontsize=12)
    plt.title('Top Percentage Apps', fontsize=20)
    plt.tight_layout()
    top_apps = pd.DataFrame(a_count[a_count > 1000]).index.tolist()
    a_df = df[df['appName'].isin(top_apps)]
    plt.figure(figsize=(10, 5))
    sns.countplot(x='appName', data=a_df)
    plt.xticks(rotation=90)
    plt.title("Top Applications", fontsize=30)
    plt.show();
    v_count = df.groupby("Vendor")["appName"].count()
    c = v_count[v_count > 2000].sort_values().plot(kind='barh', figsize=(15, 10), color='#619CFF', zorder=2)
    c.set_ylabel('Vendors')
    plt.title('Top Vendors for Cyber Security/Other Firmwares', fontsize=15)
    plt.show();
    # Countplot
    top_appName = pd.DataFrame(v_count[v_count > 2000]).index.tolist()
    v_df = df[df['appName'].isin(top_apps)]
    plt.figure(figsize=(15, 5))
    sns.countplot(x='appName', data=v_df)
    plt.xticks(rotation=90)
    plt.title("Top apps", fontsize=10)
    plt.show();
    # Countplot
    top_vendors = pd.DataFrame(v_count[v_count > 2000]).index.tolist()
    v_df = df[df['Vendor'].isin(top_vendors)]
    plt.figure(figsize=(15, 5))
    sns.countplot(x='Vendor', data=v_df)
    plt.xticks(rotation=90)
    plt.title("Top Vendors", fontsize=20)
    plt.show();
    # Pie Chart
    top_vendors = pd.DataFrame(v_count[v_count > 5000]).index.tolist()
    v_df = df[df['Vendor'].isin(top_vendors)]
    plt.figure(figsize=(6, 4))
    v_df['Vendor'].value_counts()[::-1].plot(kind='pie', title=c, autopct="%.1f%%", fontsize=12)
    plt.title('Top Percentage Vendors', fontsize=20)
    plt.tight_layout()
    # Countplot
    top_appName = pd.DataFrame(v_count[v_count > 2000]).index.tolist()
    v_df = df[df['appName'].isin(top_apps)]
    plt.figure(figsize=(15, 5))
    sns.countplot(x='appName', data=v_df)
    plt.xticks(rotation=90)
    plt.title("Top apps", fontsize=10)
    plt.show();
    t_count = df.groupby("appName")["Version"].count()
    t_count.sort_values().plot(kind='barh', figsize=(6, 4), color='#619CFF', zorder=4)
    plt.title('App', fontsize=20)
    plt.show();
    # Scatterplot
top_vendors = pd.DataFrame(v_count[v_count > 500000]).index.tolist()
v_df = df[df['Vendor'].isin(top_vendors)]
plt.figure(figsize=(35, 25))
sns.scatterplot(x='Vendor', data=v_df)
plt.xticks(rotation=90)
plt.title("Top Vendors", fontsize=20)
plt.show();
# Scatterplot
top_appName = pd.DataFrame(v_count[v_count>200000]).index.tolist()
v_df = df[df['appName'].isin(top_appName)]
plt.figure(figsize=(55, 45))
sns.scatterplot(x='appName', data=v_df)
plt.xticks(rotation=90)
plt.title("Top apps", fontsize = 20)
plt.show();
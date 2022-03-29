import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



df=pd.read_csv('visualization.csv')
df.shape

categorical_features = df.describe(include=['object','string']).columns
categorical_features


for i in categorical_features:
    a=df[i].value_counts().nlargest(50)
    print(a)


#pie chart for top 50 in all categorical features

for feature in categorical_features[1:]:
    colors=sns.color_palette('pastel')[0:50]
    df[feature].value_counts(normalize=True).nlargest(50).plot.pie(figsize=(15, 10),colors=colors,autopct='%.2f',shadow=True)
    plt.xticks(rotation='vertical')
    plt.show()

#barplot  for top 50 in all categorical features
for i in categorical_features[1:]:
    df1=df[i].value_counts()[:50]
    plt.figure(figsize=(15, 10))
    sns.barplot(df1.index,df1.values,alpha=0.8)
    plt.ylabel('count',fontsize=12)
    plt.xlabel(i,fontsize=12)
    plt.xticks(rotation='vertical')
    plt.show()

df_cpe=pd.DataFrame( df.groupby(['Type','Vendor','appName'])['CPEMatchString'])
df_cpe

cpematch_count = df.groupby("Type")['CPEMatchString'].count()
cpematch_count.sort_values().plot(kind='barh',figsize=(10, 6), color=colors)
plt.title('Type of Product', fontsize=20)
plt.show()








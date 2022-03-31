import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
os.chdir('lakshmideepthi/Visualization')
print(os.getcwd())
df=pd.read_csv("../../lakshmideepthi/Visualization/visualization.csv",encoding="utf-8")
df=df.drop(["Unnamed: 0"],axis=1)
df.head()
# x=df.info()
for c in df.columns.tolist():
    print('--'*50,f'\nColumn :{c}\n')
    print(df[c].value_counts())

df.isnull().sum()

len(df["appName"].unique())
count1=df.appName.value_counts

len(df["Vendor"].unique())
count2=df.Vendor.value_counts

len(df["Version"].unique())
count3=df.Version.value_counts

len(df["CPEMatchString"].unique())
count4=df.CPEMatchString.value_counts


### VISUALIZATIONS
# Countplot
def count_plot(*args):
    plt.figure(figsize=(10,25))
    y1=sns.countplot(y=df["Vendor"],orient="h")
    return y1
count_plot


def count_plot(*args):
    plt.figure(figsize=(10,25))
    y1=sns.countplot(y=df["Vendor"],orient="h")
    return y1
count_plot
# Barplot
count = df.groupby("Vendor")["appName"].count()
c = count[count>2000].sort_values().plot(kind='barh',figsize=(15, 10), color='#619CFF', zorder=2)
c.set_ylabel('Vendors')
plt.title('Top Vendors for Cyber Security/Other Firmwares', fontsize=15)
plt.show()

# pie chart
top_vendors = pd.DataFrame(count[count>5000]).index.tolist()
v_df = df[df['Vendor'].isin(top_vendors)]
plt.figure(figsize=(10,8))
v_df['Vendor'].value_counts()[::-1].plot(kind='pie', title=c, autopct="%.1f%%", fontsize=12)
plt.title('Top Percentage Vendors', fontsize=20)

plt.tight_layout()

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
os.chdir('lakshmideepthi/Visualization')
print(os.getcwd())
df=pd.read_csv("../../../../visualization.csv",encoding="utf-8")
df.head()
# # x=df.info()

# df1=df.drop(["Unnamed: 0"],axis=1)

# len(df["appName"].unique())
# count1=df.appName.value_counts

# len(df["Vendor"].unique())
# count2=df.Vendor.value_counts

# len(df["Version"].unique())
# count3=df.Version.value_counts

# len(df["CPEMatchString"].unique())
# count4=df.CPEMatchString.value_counts


# ### VISUALIZATIONS

# def count_plot(*args):
#     plt.figure(figsize=(10,25))
#     y1=sns.countplot(y=df["Vendor"],orient="h")
#     return y1
# count_plot
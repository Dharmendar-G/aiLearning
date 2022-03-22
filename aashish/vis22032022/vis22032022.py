

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from plotly.offline import iplot

df=pd.read_csv("/content/visualization.csv")

df.shape

df.head()

x=list(df['Type'].unique())
y=[]
for i in df['Type'].value_counts():
  y.append(i)
plt.bar(x, y)
for i in range(len(x)):
       plt.text(i, y[i], y[i], ha = 'center',
                 Bbox = dict(facecolor = 'red', alpha =.5))
       plt.title("Types")
plt.show()

trace = go.Pie(labels = x, values = y)
data = [trace]
fig = go.Figure(data = data)
print("\t\t\t\t\t\t\tTypes")
iplot(fig)

for n in ["Vendor","appName"]:
    lis=list(df[n].unique())
    l=list(filter(lambda x:type(x)==str and len(x)<=7 and len(x)>=5,lis))
    dic={}
    v=list(df[n])
    for i in l:
      dic[i]=v.count(i)
    dic=sorted(dic.items(),key=lambda x:x[1],reverse=True)
    dic=dic[0:10]
    x=[]
    y=[]
    for (i,j) in dic:
      x.append(i)
      y.append(j)
    plt.bar(x, y)
    plt.title(f"Top 10 {n}s")
    for i in range(len(x)):
          plt.text(i, y[i], y[i], ha = 'center',
                    Bbox = dict(facecolor = 'red', alpha =.6))
    plt.xticks(range(len(x)), x, rotation='vertical')
    plt.show()

for z in df['Type'].unique():
    d=df.loc[df['Type'] ==z]
    lis=list(d[n].unique())
    l=list(filter(lambda x:type(x)==str and len(x)<=7 and len(x)>=5,lis))
    dic={}
    v=list(d[n])
    for i in l:
      dic[i]=v.count(i)
    dic=sorted(dic.items(),key=lambda x:x[1],reverse=True)
    dic=dic[0:10]
    x=[]
    y=[]
    for (i,j) in dic:
      x.append(i)
      y.append(j)
    plt.bar(x, y)
    plt.title(f"Top 10 App Names in Type {z}")
    for i in range(len(x)):
          plt.text(i, y[i], y[i], ha = 'center',
                    Bbox = dict(facecolor = 'cyan', alpha =.8))
    plt.xticks(range(len(x)), x, rotation='vertical')
    plt.show()
    plt.title(f"Top 10 App Names of Type {z}")
    plt.pie(y,labels=x,
    autopct = '%1.1f%%',radius=2)
    plt.show()

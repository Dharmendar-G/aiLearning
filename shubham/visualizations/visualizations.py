import pandas as pd
import glob
import matplotlib.pyplot as plt
import zipfile
import os
print(os.getcwd())
#
# unzipfile = zipfile.ZipFile("visualization.zip")
#
# zipcsv = unzipfile.open('visualization.csv')
#
# df = pd.read_csv(zipcsv)
#
# print(df)
#
# plt.bar(df["Type"].unique(),df['Type'].value_counts())
# plt.show()
#
# plt.pie(df['Type'].value_counts(),labels=df["Type"].unique())
# plt.show()
#
# print("shubham ploting")

# files = os.listdir("./aiLearning/dataset")
# csv_files = list(filter(lambda f: f.endswith('.csv'), files))

for p,n,f in os.walk(os.getcwd()):
    for a in f:
        a = str(a)
        if a.endswith('.csv'):
            print(a)
            print(p)

print("how to work with multiple csv with multiple folder.")

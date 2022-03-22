import csv
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
# path = 'C:/FileLocation/'
# file = 'filename.csv'
f = open("SivaShankar/all_csv_to_standardTemplate/all.csv",'rt')
reader = csv.reader(f)

#each row of excel is taken into the list of rows => [[row1],[row2]....]
csv_list = []
for l in reader:
    csv_list.append(l)
f.close()
#print(csv_list)

#now pandas has no problem getting into a df
df = pd.DataFrame(csv_list)
print(df.head(3))

a,b = df.shape
# print(a,b)
# print(a//2)

stdf = pd.read_csv("dataset/StandardTemplate.csv", encoding='cp1252')
print(stdf.shape)

aa = stdf.shape[0]
if aa!=a//2 :
    for x in range(a//2):
        stdf.loc[x] = np.nan

print(stdf.shape)

print(stdf.head())

for r in range(0,a-1,2):
    stdf["fileName"].iloc[r//2] = r//2
    for c in range(0,b):
        if df.iloc[r][c] in list(stdf.columns) :
            stdf[df.iloc[r][c]].iloc[r//2] = df.iloc[r+1][c]

print(stdf.head(5))

stdf.to_csv("SivaShankar/all_csv_to_standardTemplate/all_standard_template_update.csv",index=False)
print("Done....!!! \n File Saved!")
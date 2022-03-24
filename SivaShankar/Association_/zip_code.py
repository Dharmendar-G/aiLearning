import os
import pandas as pd
import zipfile
import re
import time


path = f"{os.getcwd()}\dataset\\associationRuleMining.zip"
zf = zipfile.ZipFile(path)

start = time.time()
with zf as thezip:
    a = thezip.infolist() # list 
    b = [pd.read_csv(thezip.open(a[x].filename,mode='r')) for x in range(1,len(a))]
    fdf = pd.concat(b)
end = time.time()
# Return the current time in seconds 
print(f"Total number of csv files  :  {len(a)}\n")
print(f"time taken to read the files and make them to a dataframe: {end-start} sec \n ")

c = [x for x in fdf.columns if bool(re.search("Unnamed:",x))]

fdf.drop(c,axis=1,inplace=True)

print(fdf.head(3))

print(f"shape (rows,columns) of fdf dataframe (the concat of all {len(a)-1} csv files) : {fdf.shape} \n")

print(f" List of column names in final dataframe {fdf.columns}\n")
#print(type(zf))
#print(zf.infolist())

# for f in zf.infolist():
#     c =1
#     print(f.filename)
#     df = pd.read_csv(zf.open(f.filename))
#     print(df.head(3))
#     if c==4:
#         break

#print(zf.infolist()[1].filename)


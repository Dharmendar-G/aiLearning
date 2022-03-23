import zipfile 
import pandas as pd
import os
import re
import time
path = f"{os.getcwd()}\dataset\\associationRuleMining.zip"
zf = zipfile.ZipFile(path)
start = time.time()
with zf as thezip:
        a= thezip.infolist() # list
        b= [pd.read_csv(thezip.open(a[x].filename,mode='r')) for x in range(1,len(a))]
        finaldf = pd.concat(b)
end =time.time()
print(f"Total number of csv files :{len(a)}\n")
print(f"Time taken to read the files and make them into a dataframe : {end-start} sec \n")
c= [x for x in finaldf.columns if bool (re.search("Unnamed:",x))]
finaldf.drop(c,axis=1,inplace=True)
print(finaldf.head())
print(f"shape (rows,columns) of fdf dataframe (the concat of all {len(a)-1} csv files) : {finaldf.shape} \n")

print(f" List of column names in final dataframe {finaldf.columns}\n")
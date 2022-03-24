import pandas as pd
import os
import zipfile
import re
import time

#z = zipfile.ZipFile('associationRuleMining.zip')
os.chdir('SivaShankar/Association_')
print(os.getcwd())
with zipfile.ZipFile('../../dataset/associationRuleMining.zip') as thezip:
    a = thezip.infolist()# list 
    #print(a)
    d = len(a)-1 # 25000 files
    print("no of csv files : ", d)
    no_of_batches = round(d/500) # 500 files per batch => 50b * 500 files = 25000 files
    print("total number of batches ",no_of_batches)

    s = 1
    c = 1
    no_of_files_batch = 500 # no of files per batch
    bdf = [] # list of batch dfs=> [[b1],[b2],......]
    max_range = len(a)
    min_range = 1 # as the  0 th index has the folder path 
    for x in range(min_range,max_range,no_of_files_batch):

        g = (s*(no_of_files_batch)+1 )

        try:
            bdf.append([pd.read_csv(thezip.open(a[y].filename,mode='r')) for y in range(x,g)])
        except IndexError:
            g = max_range
            bdf.append([pd.read_csv(thezip.open(a[y].filename,mode='r')) for y in range(x,g)])
            

        s+=1 
        c+=1
        
cou = 1
freq = {}

'''x = 1 => first batch => has list of 500 dfs
pbdf for first instance => the df for first batch

final => stores all the pbdf generated in each batch as list of pbdfs
'''
final = [ ]
for x in bdf:
    pbdf = pd.concat(x)
    #display(pbdf)
    for x in range(len(pbdf.vulnerabilities)):
        sg = pbdf.vulnerabilities.iloc[x].strip('[]').split(',')

        for x in sg:
            y = x.strip(" ''")
            if y in freq.keys():
                freq[y] +=1
            else:
                freq[y] = 1
    
    final.append(pbdf)
        
# generates overall df for 25000 csvs
fdf = pd.concat(final)

print("columns in the dataframe\n ",fdf.columns)

# to remove the columns that match with the Unnamed:
rem = []
for x in fdf.columns:
    if re.search("Unnamed:",x):
        rem.append(x)

fdf.drop(rem,axis=1,inplace = True)

print("columns in dataframe after removing the unwanted columns : \n",fdf.columns)


# to create the new columns with list of freqs of the related vulns of cveid's
new_col = []
for x in range(len(fdf.vulnerabilities)):
    sg = fdf.vulnerabilities.iloc[x].strip('[]').split(',')
    sg = [x.strip("'' ") for x in sg]
    cn = []
    for x in sg:
        cn.append(freq[x])
    new_col.append(cn)
    
    
# adding the new_col to the end of the dataframe fdf.
fdf_n = fdf.assign(freq_vulns = new_col)

print(fdf_n.head())

fdf.to_csv("associate_csv_update.csv")

print("Done Saving the update.....!!!!")


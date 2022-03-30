#Task
#write a program to find index number, based on the following conditions:
# 1.Unzip indexNumber.zip from dataset folder.
# 2.Pick one key -cpe23Uri from sourceDF.csv
# 3Find the indexNumber of the above key from “CPEMatchString” columns of TargetDf.csv
import os
import re
import zipfile
import pandas as pd
#print(os.listdir())

# Unzipping the files
with zipfile.ZipFile('../../dataset/indexNumber.zip', 'r') as zf:
    zf.extractall(os.getcwd())
#print(os.listdir())

# Datasets source and target
s_df = pd.read_csv('sourceDF.csv')
s_df = s_df.drop('Unnamed: 0', axis=1)
t_df = pd.read_csv('TargetDF.csv', nrows=19, low_memory=False)
t_df = t_df.drop('Unnamed: 0', axis=1)

# function for preprocessing cleaning the irrelevant part extract the common part in source data (-cpe23Uri
def s_preprocess(text):
    s1 = "".join(text.split('\\'))
    s2 = "".join(s1.split('/'))
    s3 = re.sub('2.3:', '', s2)
    s4 = re.sub('\@[a-zA-Z0-9._]+','', s3)
    s5 = re.sub('\:[*]+', '', s4)
    s = re.sub('\$[a-zA-Z0-9._]+', '', s5).strip()
    return s.lower()
#print(s_preprocess(s_df['cpe23Uri'].tolist()[0]))

# function for preprocessing cleaning the irrelevant part extract the common part in Target data(“CPEMatchString” )
def t_preprocess(text):
    s1 = "".join(text.split('\\'))
    s2 = "".join(s1.split('/'))
    s3 = re.sub('2.3:', '', s2)
    s4 = re.sub('\%[a-zA-Z0-9._]+', '', s3)
    s5 = re.sub('\:[~]+', '', s4)
    s = re.sub('~','', s5).strip()
    return s.lower()
# print(t_preprocess(t_df['CPEMatchString'].tolist()[0]))

# Taking a key from source_df
key = s_preprocess(s_df['cpe23Uri'].tolist()[6])

# Searching for match in target_df
match_index = []
for i in range(len(t_df['CPEMatchString'].tolist())):
    s = t_preprocess(t_df['CPEMatchString'].tolist()[i])
    if key == s:
        print(f"match at index : {i}")
        match_index.append(i)
    else:
        continue

# now matching all keys return index number
for i,c in enumerate(t_df['CPEMatchString'].tolist()):
    print(f"index: {i} : {t_preprocess(c)}")
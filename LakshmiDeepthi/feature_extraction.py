'''
Task-Finding the key value from given datasets.
Given keys:
  1. State
  2. ZIP
  3. Date Of Construction
  4. Building or A&A 
  5. contents
  6. priorloss
  7. Building or A&A 
  8. Deductible(not applicable to excess)
  9. Contents
Findings
I found that keys are present in first 3 datasets.
found Values according to keys that are present in first 3 datasets.
but each key and value pattern are different from each dataset. --eg. in first dataset 'Building or A A $3949000' -- in second dataset it is like 'Building or A A S 3949000'
Steps I followed to grab the values:
loading the datsets.
i joined the text column as a string.
removed Special Chatracters and spaces between words
In each dataset i checked the key and corresponding value(Manually)
After that i tried to found a pattern for each key and value(Using Regular Expression)
i wrote code for all key and values using functions that makes easy for grabbing values from all datasets.
in some cases i did hard coding .
Updated the values in " Standard Template".
Converted that Standard template into csv file.
'''


# Importing the necessary modules

import pandas as pd
import numpy as np
import re
import os
os.chdir('lakshmideepthi/feature_extraction')
print(os.getcwd())
df=pd.read_csv('../../dataset/StandardTemplate.csv',encoding = 'cp1252')
df1 = pd.read_csv('../../dataset/1.csv',encoding = 'utf-8')

df2=pd.read_csv('../../dataset/2.csv',encoding = 'utf-8')
df3=pd.read_csv('../../dataset/3.csv',encoding = 'utf-8')
df4=pd.read_csv('../../dataset/4.csv',encoding = 'utf-8')
df5=pd.read_csv('../../dataset/5.csv',encoding = 'utf-8')
df6=pd.read_csv('../../dataset/6.csv',encoding = 'utf-8')


data1=" ".join(list(df1['text'].values))
data2=" ".join(list(df2['text'].values))
data3=" ".join(list(df3['text'].values))
data4=" ".join(list(df4['text'].values))
data6=" ".join(list(df6['text'].values)) 

# function for grabbing the values for given keys in every dataset


def cln_data(string):
    final = " ".join(re.findall(r"[a-zA-Z0-9]+", string))
    return final
data=cln_data(data6)

def remove(string):
    return "".join(string.split())
datar=remove(data)

def Zip(string):
    try:
        pattern=r"Zip.\d+"
        s=re.findall(pattern,string)
        num1 = re.findall(r'\d+',s[0]) 
        return num1
    except  IndexError:
        print("ZIP is not presented in this dataset")

def DateofConstruction(string):
    try:
        pattern=r"DateofConstruction.\d+"
        t=re.findall(pattern,string)
        num1 = re.findall(r'\d+',t[0]) 
        return num1
    except IndexError:
        print(" DOFC is nor presented in this dataset")
        
def State(string):
    try:
        pattern=r"State.\w"
        u=re.findall(pattern,string)
        return u[0][5::]
    except IndexError:
        print("state is not presented in this dataset")

def Building1(string):  
    try:
        pattern=r"BuildingorAA..\d+"
        z=re.findall(pattern,string)
        num2 = re.findall(r'\d+',z[0])
        num3 = re.findall(r'\d+',z[1])
        return num2,num3
    except IndexError:
            print(" Building or AA is presented in different pattern in this dataset")
            
            
def Building2(string):
    try:
        pattern=r"BuildingorAX.....\d+"
        z=re.findall(pattern,string)
        num = re.findall(r'\d+',z[0]) 
        num2 = re.findall(r'\d+',z[1]) 
        return num,num2
    except IndexError:
        print("Building or AA is presented in different pattern in this dataset")
        
def Contents(string):
    try:
        pattern=r"Contents.\d+"
        a=re.findall(pattern,string)
        num1 = re.findall(r'\d+',a[0])
        num2 = re.findall(r'\d+',a[1]) 
        return num1,num2
    except:
        print(" Contents are not presented in this dataset")

def Deductible(string):
    try:
        pattern=r"Deductible.\d+"
        x=re.findall(pattern,string)
        num = re.findall(r'\d+',x[0]) 
        return num
    except IndexError:
        print("Deductible is not presented in this dataset")


def Deductible1(string):
    try:
        pattern=r"DeductibleNotapplicabletoexcess.\d+"
        x=re.findall(pattern,string)
        num = re.findall(r'\d+',x[0]) 
        return num
    except IndexError:
        print("DeductibleNotapplicabletoexcess not presented in given dataset ")
        

d=Zip(datar)
d2=DateofConstruction(datar)
d3=State(datar)
d4=Building1(datar)
# d4[0]
# # d4[1]
d7=Building2(datar)
d5=Contents(datar)
# d5[0]
# d5[1]
d6=Deductible(datar)
# d6[0]
# d6[1]
d8=Deductible1(datar)

df.at[3,"Zip.1"]  =d      # this df.at[row index,column name]=value will update the value at particular location

dfu=df.to_csv("Submission.csv",index=False)
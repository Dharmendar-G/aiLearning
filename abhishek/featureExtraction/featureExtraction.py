import numpy as np
import pandas as pd
import re
import os
import nltk


df1 = pd.read_csv('aiLearning\dataset\1.csv')
df2 = pd.read_csv('aiLearning\dataset\2.csv')
df3 = pd.read_csv('aiLearning\dataset\3.csv')
df4 = pd.read_csv('aiLearning\dataset\4.csv')
df5 = pd.read_csv('aiLearning\dataset\5.csv')
df6 = pd.read_csv('aiLearning\dataset\6.csv')
df7 = pd.read_csv('aiLearning\dataset\StandardTemplate.csv')

data1 = []
data2 = []
data3 = []
data4 = []
data5 = []
data6 = []
for i in df1["new_col"].astype(str):
    data1.append(i)
data1=list(set(data1))

for i in df2["new_col"].astype(str):
    data2.append(i)
data=list(set(data2))

for i in df3["new_col"].astype(str):
    data3.append(i)
data3=list(set(data3))

for i in df4["new_col"].astype(str):
    data4.append(i)
data4=list(set(data4))

for i in df6["new_col"].astype(str):
    data6.append(i)
    
for i in df5["new_col"].astype(str):
    data5.append(i)
data5=list(set(data5)) 

#to get date of loss
def date_of_loss(text):
    match = re.findall(r'(Date of Loss:)\s+(\d{2}/\d{2}/\d{4})',str(text))
    return match

#no of stories
def no_of_stories(row_no):
    list2=[]
    data1 = ' '.join(list(df1['text'].values))
    data1 = nltk.word_tokenize(data1.lower())
    for i in range(len(data1)):
        if bool(re.search(":",data1[i])):
            if bool(re.search("stories",data1[i-1])):
                list2.append(data1[i+1])
    #print(list2)
    df7.loc[row_no,'No of Stories'] = list2[i]

#Date of Construction
def get_date_of_con(dfNo):
    list2=[]
    data1 = ' '.join(list(df1['text'].values))
    data1 = nltk.word_tokenize(data1.lower())
    for i in range(len(data1)):
        if bool(re.search(":",data1[i])):
            if bool(re.search("construction",data1[i-1])):
                l = ""
                for j in range(i+1,i+4):
                    l+=str(data1[j])
                list2.append(l)    
    #print(list2)
    df7.loc[dfNo,'Date of Construction'] = list2[i]


#csv1
for i in df1["new_col"].astype(str):
    data1.append(i)
data1=list(set(data1))

def city1(text):
    city=re.findall(r'(City:)\s+([a-zA-Z]+)',str(text))
    return city[0][1]

def country_req1(text):
    country=re.findall(r'(County [(]required[)]\S+\s([a-zA-Z]+))',str(text))
    return country[0][1]

df7.loc[0,'City'] = city1(data1)
#df7.loc[0,'Street'] = city1(daat1)[0][1].split(',')[1]
#df7.loc[0,'Zip'] = city1(daat1)[0][2]
df7.loc[0, 'County (required)'] = country_req1(data1)

#csv2
for i in df2["new_col"].astype(str):
    data2.append(i)
data2=list(set(data2))

def city2(text):
    city=re.findall(r'(City:)\s+([a-zA-Z]+\s+[a-zA-Z]+)',str(text))
    return city[0][1]
def country_req(text):
    country=re.findall(r'(Counntty [(]required[)])\S([a-zA-z].+)',str(text))
    return country[0][1].split(":")[0]

df7.loc[1,'City'] = city2(data2)
#df7.loc[1,'Street'] = city2(data2)[0][1].split(',')[1]
#df7.loc[1,'Zip'] = city2(data2)[0][2]
df7.loc[1, 'County (required)'] = country_req(data2)

#csv3
for i in df3["new_col"].astype(str):
    data3.append(i)
data3=list(set(data3))

def city3(text):
    city=re.findall(r'(City:)\s+([a-zA-Z]+\s+[a-zA-Z]+)',str(text))
    return city[0][1]

def country_req(text):
    country=re.findall(r'(County: (requirep) __)\s+([a-zA-z].+)',str(text))
    return country

df7.loc[2,'City'] = city3(data3)
#df7.loc[2,'Street'] = city1(r)[0][1].split(',')[1]
#df7.loc[2,'Zip'] = city1(r)[0][2]
df7.loc[2, 'County (required)'] = country_req(data3)

#csv4
for i in df4["new_col"].astype(str):
    data4.append(i)
data4=list(set(data4))

def city4(text):
    city=re.findall(r'(City/State/Zipcode)\s+([a-zA-Z]+\S+\s+[a-zA-Z]+\S+\s+([0-9]+))',str(text))
    return city

df7.loc[3,'City'] = city4(data4)[0][1].split(',')[0]
df7.loc[3,'Street'] = city4(data4)[0][1].split(',')[1]
df7.loc[3,'Zip'] = city4(data4)[0][2]
#df7.loc[3, 'County (required)'] = 

#csv5
for i in df5["new_col"].astype(str):
    data5.append(i)
data5=list(set(data5))

def city5(text):
    #city=re.findall(r'(City / State / Zipcode)\s+([a-zA-Z]+\s+[a-zA-Z,]+\s([a-zA-Z]+\s([a-zA-z]+))+)',str(text))
    city = re.findall(r'(City / State / Zipcode)\s+([a-zA-Z]+\s+[a-zA-Z,]+\s([a-zA-Z]+\s([a-zA-z]+))+\S+\s+([0-9]+))',str(text))
    #return city[0][1].split(",")
    return city

df7.loc[4,'City'] = city5(data5)[0][1].split(',')[0]
df7.loc[4,'Street'] = city5(data5)[0][1].split(',')[1]
df7.loc[4,'Zip'] = city5(data5)[0][4]
#df7.loc[4, 'County (required)'] = 
df7.loc[4, 'Date of Loss'] = date_of_loss[0][1]

#csv6
for i in df6["new_col"].astype(str):
    data6.append(i)

def city6(text):
    city=re.findall(r'(City/State/Zipcode:)\s+([a-zA-Z]+\s+[a-zA-Z,]+\s([a-zA-Z]+))',str(text))
    #city=re.findall(r'(City/State/Zipcode:)\s+([a-zA-Z]+\S+\s+[a-zA-Z]+\S+\s+([0-9]+))',str(text))
    #return city[0][1].split(",")[0]
    return city[0]

def county_req(text):
    country=re.findall(r'(County: (requirep) __)\s+([a-zA-z].+)',str(text))
    return country

df7.loc[5,'City'] = city6(data6)[0][1]
df7.loc[5, 'Street'] = city6(data6)[0][1]
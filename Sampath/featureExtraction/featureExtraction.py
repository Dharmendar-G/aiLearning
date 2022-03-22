## Given keys:
      #1. Slab on Grade
      #2. Crawl Space
      #3. Elevated (If yes)
      #4. Piers/Posts/Columns 
      #5. Flood Zone
      #6. BFE
      #7. LFE 
      #8. Date quote submitted to WNC
      #9. Policy Form


import pandas as pd
import numpy as np
import re

e1=pd.read_csv('1.csv')
e2=pd.read_csv('2.csv')
e3=pd.read_csv('3.csv')
e4=pd.read_csv('4.csv')
e5=pd.read_csv('5.csv')
e6=pd.read_csv('6.csv')

files=[e1,e2,e3,e4,e5,e6]

e1.head(3)

e1.info()

for i,j in enumerate(e1.text):
    print(i,j)

## Function to find "Date quote submitted to WNC" in the dataset
## They are only available in 1,2,3 Datasets

def date_check(df):
    solution=[]
    try:
        for i in range(len(df['text'])):
            if bool(re.search('WNC:',df['text'][i])):
                #print(df['text'][i+1])
                solution.append(df['text'][i+1])
    except Exception as e:
        solution='0'
        
    return solution

date_check(e1)

date_check(e2)

date_check(e3)

date_check(e4)

date_check(e6)

datecheck=[]
for i in files:
    datecheck.append(date_check(i))

datecheckk=['91/14/22','91/12/2022','12/7/21','0','0','0']

datecheckk

## Function to find "Slab on grade" or "Crawl Space" in a dataset 
## These are available in only 1,2,3 datasets

def GradeOrSpace(df):
    solution2=[]
    try:
        for i in range(len(df['text'])):
            if bool(re.search('Grade',df['text'][i])):
                g=df['conf'][i]
                #print(g)
            elif bool(re.search('Space',df['text'][i])):
                s=df['conf'][i]
                #print(s)  
        if g>s:
            solution2='1'
        else:
            solution2='2'
            
    except Exception as e:
        solution2='0'
        
    return solution2
    
#SLAB ON GRADE 1
#CRAWL SPACE 2
# else 0

GradeOrSpace(e1)

GradeOrSpace(e2)

GradeOrSpace(e3)

GradeOrSpace(e4)

GradeOrSpace(e5)

GradeOrSpace(e6)

GorP=[]
for i in files:
    GorP.append(GradeOrSpace(i))

GorP



## Function to find FLOOD ZONE, BFE, LFE
## Present in only Datasets 1,2

def BFEorLFE(df):
    solution3=[]
    try:
        for i in range(len(df['text'])):
            if bool(re.search('Flood',df['text'][i])):
                if bool(re.search('Zone,',df['text'][i+1])):
                    for j in range(1,10):
                        if df['text'][i+j]=='BFE':
                            BVal=df['conf'][i+j]
                        elif df['text'][i+j]=='LFE':
                             FVal=df['conf'][i+j]
                    
        if BVal>FVal:
            solution3='1'
        else:
            solution3='2'
    
    except Exception as e:
        solution3='0'
        
    return solution3
    
#BFE 1
#LFE 2
#EMPTY 0

BFEorLFE(e1)

BFEorLFE(e2)

BFEorLFE(e3)

BFEorLFE(e4)

BFEorLFE(e5)

BFEorLFE(e6)

BorL=[]
for i in files:
    BorL.append(BFEorLFE(i))

BorL

## Function to find Piers/Posts/Columns if elevated

## The below function will work on only Dataset1
### for other datasets the information is not well, so it is hard to extract the data.

def ElevatedPPC(df):
    solution4=[]
    try:
        for i in range(len(df['text'])):
            if bool(re.search('If',df['text'][i])):
                if bool(re.search('elevated,',df['text'][i+1])):
                    for j in range(1,10):
                        if df['text'][i+j]=='OpPiers':
                            num1=df['conf'][i+j]
                            #print(num1)
                        elif df['text'][i+j]=='OPosts':
                            num2=df['conf'][i+j]
                            #print(num2)
                        elif df['text'][i+j]=='Ocolumns':
                            num3=df['conf'][i+j]
                            #print(num3)
                    
        if (num1 > num2) and (num1 > num3):
            largest = num1
            solution4='2'
        elif (num2 > num1) and (num2 > num3):
            largest = num2
            solution4='3'
        else:
            largest = num3
            solution4='1'
    
    except Exception as e:
        solution4='0'
        
    return solution4

#columns 1
#piers 2
#posts 3
#empty 0

ElevatedPPC(e1)

ElevatedPPC(e2)

ElevatedPPC(e3)

ElevatedPPC(e4)

ElevatedPPC(e5)

ElevatedPPC(e6)

elevated=[]
for i in files:
    elevated.append(ElevatedPPC(i))

elevated

### From e1 dataset the we can say that
##### It is elevated with COLUMNS

StandardTemplate=pd.read_csv('StandardTemplate.csv',encoding='cp1252')

StandardTemplate.head(7)

for i in enumerate(StandardTemplate.columns):
    print(i)

StandardTemplate.iloc[:,74:82]

## Outputs from the Functions defined

Slab_on_Grade = [0,0,0,0,0,0]
Crawl_Space = [1,1,1,0,0,0]
Elevated = [1,0,0,0,0,0]
Piers_Posts_Columns = [1,0,0,0,0,0]
FloodZone = [1,1,0,0,0,0]
BFE = [1,1,0,0,0,0]
LFE = [0,0,0,0,0,0]
Date = ['91/14/22', '91/12/2022', '12/7/21', '0', '0', '0']

## Append them in the respective Columns

StandardTemplate['Date Quote Submitted to WNC']=Date
StandardTemplate['LFE.1']=LFE
StandardTemplate['BFE.1']=BFE
StandardTemplate['Flood Zone.2']=FloodZone
StandardTemplate['Piers/Posts/Columns (circle)']=Piers_Posts_Columns
StandardTemplate['Elevated (If Yes by)']=Elevated
StandardTemplate['Crawl Space']=Crawl_Space
StandardTemplate['Slab on Grade']=Slab_on_Grade

StandardTemplate.iloc[:,74:82]

StandardTemplate.to_csv('submission.csv')
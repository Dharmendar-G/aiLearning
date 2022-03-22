#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
warnings.filterwarnings("ignore")

def condo(token):
    l = len(token)
    for i in range(l):
        if bool(re.search('condo/cooperative',token[i])):
            if token[i-1]=='©':
                return "N"
                break
            if token[i-1]=='o':
                return "N"
                break
        if bool(re.search('condominium/cooperative',token[i])):
            if token[i+1]=='x':
                return "N"
                break
        if bool(re.search('condominium',token[i])):
            if token[i+1]=='o':
                return "N"
                break
            if token[i+1]=="policy":
                return "N"
                break

    

def family_2(token):
    l = len(token)
    for i in range(l):
    #     if '2-family' not in lst1:
    #         print("Nan")
    #         break
        if bool(re.search('2-family',token[i])):
            if token[i-1]=='©':
                return "N"
                break
            if token[i-1]=='o':
                return "N"
                break
            if token[i+1]=='condominium/cooperative':
                return "N"
                break
        if bool(re.search('2-4',token[i])):
            if token[i+1]=='family' and token[i-1]=='o':
                return "N"
            if token[i+1]=='family' and token[i-1]=='|_|':
                return "N"
            if token[i+1]=='family' and token[i+1]=='o':
                return "N"


def single_family(token):
    l = len(token)
    for i in range(l):
#        
        
      
        if bool(re.search('single-family',token[i])):
            if token[i-1]=='©':
                return "Y"
                break
        elif 'single-family'not in token:
            
            
            if bool(re.search('single',token[i])):
                if token[i+1]== 'family' and token[i+2]=='xs':
                    return "N"
                    break
                elif token[i+1]== 'family' and token[i+2]=='x':
                    return "N"
                    break
                elif token[i+1]== 'family' and token[i+2]=='o':
                    return "N"
                    break
            if 'single' not in token:
                return "Nan"
                break

def storey(token):
    l=len(token)
    for i in range(l):
        if bool(re.search("stories",lst1[i])):
            if lst1[i+2].isnumeric():
                return lst1[i+2]
            elif bool(re.search("__3.__",lst1[i-3])):
                return lst1[i-3].strip("_.")
            else:
                return "Nan"
        elif "stories" not in token:
            return "Nan"
            break

def basement(token):
    l = len(token)
    for i in range(l):
        if "obasement" in token:
            return "N"
            break

        if bool(re.search("basement",token[i])):
            if token[i-1]=="o":
                return "N"
            elif token[i-1]=="©":
                return "Y"
            elif token[i-2]=="___y":
                return "Y"
            elif token[i-1]=="_|":
                return "Nan"
        if "basement" not in token:
            return "Nan"
            break


def finished(token):
    l = len(token)
    for i in range(l):
        if "ofinished" in token:
            return "N"
            break
        if "©finished" in token:
            return "Y"
            break
        if bool(re.search("finished",token[i])):
            if token[i-1]==')':
                return "Y"
            if token[i-1]=='o':
                return "N"
                break
            if token[i+1]=="|¥":
                return "Y"
            if token[i+1]=='o':
                return "N"
                break

def flood_zone(token):
    l = len(token)
    for i in range(l):
        if bool(re.search("zone",token[i])) and token[i+1]==":" :
        
            if token[i+2]=="x" or token[i+2]=="__x":
                return "X"
                break
            else:
                return "Nan"
                break
        elif bool(re.search("zone",token[i])):
            if token[i+1]=="———x_~——sstfype":
   
                return "X"
                break
        if "zone" not in token:
            return "Nan"
            break                      

def unfinished(token):
    l = len(token)
    for i in range(l):
        if 'ounfinished' in token:
            return "N"
            break
        if bool(re.search("unfinished",token[i])):
            if token[i-1]=='___x___':
                return "N"
                break
            if token[i+1]=='o':
                return "N"
                break
            if token[i+1]=='[':
                return "N"


def enclosure(token):
    l = len(token)
    for i in range(l):
        if 'oenclosure' in token:
            return 'N'
            break
        if "enclosure" not in token:
            return "Nan"
            break
        if bool(re.search("enclosure",token[i])):
            if token[i-1]=='©o':
                return"N"
                break
            if token[i+1]== 'slab' and token[i-1]== 'unfinished':
                return "N"
                break
            if token[i-1] == 'o':
                return "N"
                break
    

# reading csv file
stdf= pd.read_csv("../..dataset/StandardTemplate.csv",encoding='cp1252')

for xy in range(1,7):
    z=[]
    df = pd.read_csv(f"../..dataset/{xy}.csv")
    for y in df.columns:
        if bool(re.search("Unnamed:",y)):
            z.append(y)
    df.drop(z,axis=1,inplace=True)
    #display(df)
    ss = ""
    for x in list(df.block_num.unique()):
        data = df[df["block_num"]==x]
        for y in list(data.par_num.unique()):
            dat = data[data["par_num"]==y]
            for z in list(dat.line_num.unique()):
                da = dat[dat["line_num"]==z]
                ss +=" ".join([str(list(da.text[da.word_num == w])[0]) for w in list(da.word_num.unique())])+" "
    ss = ss.lower()
    lst1 = word_tokenize(ss)
    fz = flood_zone(lst1)
    sf = single_family(lst1)
    f2 = family_2(lst1)
    con= condo(lst1)
    st = storey(lst1)
    bm = basement(lst1)
    fin = finished(lst1)
    unfin = unfinished(lst1)
    enc = enclosure(lst1)
    #print(stdf['fileName'].iloc[xy-1])
    if stdf['fileName'].iloc[xy-1] == xy:
        stdf['Flood Zone.1'].iloc[xy-1] = str(fz)
        stdf['Type of Building : Single Family'].iloc[xy-1]=str(sf)
        stdf['Type of Building : 2-Family'].iloc[xy-1]=str(f2)
        stdf['Type of Building : Condominium\\Cooperative'].iloc[xy-1]=str(con)
        stdf['Number of Stories (excluding basement)'].iloc[xy-1]=str(st)
        stdf['Basement(Y/N)'].iloc[xy-1]=str(bm)
        stdf['Finished'].iloc[xy-1]=str(fin)
        stdf['Unfinished'].iloc[xy-1]=str(unfin)
        stdf['Enclosure'].iloc[xy-1]=str(enc)
 
stdf.to_csv("Submission.csv",index=False)


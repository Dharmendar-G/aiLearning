### Writing functions to compare each key against 6 csv files and update their respective values in the StandardTemplate.csv

## Search Keys : 

# 1. Personal (Primary) Flood or Excess Flood
# 2. Any flood loss in past 10 years? Yes/No
# 3. Elevated? (Yes/No)
# 4. If elevated, by:
# 5. Type of Building (Single-family/2-Family/Condo\Cooperative)
# 6. Foundation Information (Flood Zone/BFE/LFE)
# 7. Homeowner Policy Effective Date
# 8. Requested Effective Date of Flood Policy
# 9. Building or AA Limits

import os
import re
import pandas as pd 
import numpy
pd.set_option('display.max_columns', None)

# Standard Template Dataset
standard = pd.read_csv('C:\\Users\\DharmendraGa_5wskc\\Desktop\\AI Training\\aiLearning\\dataset\\StandardTemplate.csv', encoding='cp1252')
stdf = standard[standard.columns.tolist()[:167]]
stdf.rename(columns={'fileName':'FileName'}, inplace=True)

# Adding FileName Column with 6 empty rows
# import warnings 
# warnings.filterwarnings('ignore')
# for i in range(6):
#     stdf.loc[stdf.shape[0]] = ['-']*stdf.shape[1]
# stdf.insert(0, column='FileName', value=[x for x in range(1,7)])
# stdf

# Importing 6 matching dataframes 
df_list = []
for i in range(1,7):
    if i.endswith('csv'):
        df = pd.read_csv(f'C:\\Users\\DharmendraGa_5wskc\\Desktop\\AI Training\\aiLearning\\dataset\\{i}.csv')
        df_list.append(df)
    
# Required Keys
keys = stdf.columns.tolist()[28:37]

### key 1: 'Personal (Primary) Flood or Excess Flood'

# Checking key : Personal (Primary) Flood or Excess Flood
def PersonalOrExcessFlood(df_list):
    key = keys[0].lower().split(' or ')
    s = 'None'
    df_idx = {}
    for idx, df in enumerate(df_list, start=0):
        for b in df.block_num.unique():
            data = df[df['block_num']==b]
            blocks = list(set(data['new_col'].tolist()))
            for i in blocks:
                if '(@) '+key[0] in str(i).lower():
                    s = key[0].upper()
                    df_idx[idx+1] = s
                elif '(@) '+key[1] in str(i).lower():
                    s = key[1].upper()
                    df_idx[idx+1] = s
                else:
                    continue
    return df_idx
result = PersonalOrExcessFlood(df_list)
print(f"Dataset Matching Values for key : {keys[0]} \n\n{result}")
# Updating StandardTemplate Dataframe
for k,v in result.items():
    stdf.loc[stdf['FileName'] == k, keys[0]] = v
### key 2: 'Any flood loss in past 10 years? Yes/No'

# To check If Flood In Last 10 Years 
def FloodInLast10Years(df_list):
    key = 'in the last 10 years?'
    df_idx = {}
    for idx, df in enumerate(df_list, start=0):
        for x in list(df_list[idx].block_num.unique()):
            d = set(df_list[idx][df_list[idx]["block_num"]==x]['new_col'].tolist())
            if key in str(d).lower():
                s= str(d).split(',')
                for i in s:
                    ss = i.strip().lower()
                    if key in ss:
                        if 'oyes' and '©no' in ss:
                            df_idx[idx+1] = 'No'
                        elif '©yes' and 'ono' in ss:
                            df_idx[idx+1] = 'Yes'
                        else:
                            continue
            else:
                s = None
    return df_idx

result = FloodInLast10Years(df_list)
print(f"Dataset Matching Values for key : {keys[1]} \n\n{result}")
# Updating StandardTemplate Dataframe
for k,v in result.items():
    stdf.loc[stdf['FileName'] == k, keys[1]] = v

### key 3: 'Elevated? (Yes/No)'

# To Check if Elevated is yes or no
def IsElevated(df_list):
    key = 'elevated?'
    df_idx = {}
    for idx, df in enumerate(df_list, start=0):
        for x in list(df_list[idx].block_num.unique()):
            d = set(df_list[idx][df_list[idx]["block_num"]==x]['new_col'].tolist())
            if key in str(d).lower():
                s= str(d).split(',')
                for i in s:
                    ss = i.strip().lower()
                    if key in ss:
                        if 'oyes' and '©@no' in ss:
                            df_idx[idx+1] = 'No'
                        elif 'oyes' and 'ono' in ss:
                            df_idx[idx+1] = '-'
            else:
                s = None
    return df_idx

result = IsElevated(df_list)
print(f"Dataset Matching Values for key : {keys[2]} \n\n{result}")
# Updating StandardTemplate Dataframe
for k,v in result.items():
    stdf.loc[stdf['FileName'] == k, keys[2]] = v

### key 4: 'If elevated, by'

def IfElevatedBy(df_list):
    key = 'if elevated, by:'
    df_idx = {}
    for idx, df in enumerate(df_list, start=0):
        for x in list(df_list[0].block_num.unique()):
            d = str(set(df_list[0][df_list[0]["block_num"]==x]['new_col'].tolist())).lower()
            if key in d:
                types = ['piers', 'posts', 'columns']
                if f'©{types[0]}' in d:
                    df_idx[idx+1] = f'{types[0]}'
                elif f'©{types[1]}' in d:
                    df_idx[idx+1] = f'{types[1]}'
                elif f'©{types[2]}' in d:
                    df_idx[idx+1] = f'{types[2]}'
                else:
                    df_idx[idx+1] = '-'
            else:
                continue
    return df_idx

result = IfElevatedBy(df_list)
print(f"Dataset Matching Values for key : {keys[3]} \n\n{result}")
# Updating StandardTemplate Dataframe
for k,v in result.items():
    stdf.loc[stdf['FileName'] == k, keys[3]] = v

### key 5: 'Type of Building (Single-family/2-Family/Condo//Cooperative)'

def TypeOfBuilding(df_list):
    df_idx = {}
    for idx, df in enumerate(df_list, start=0):
        s = str(list(set(df['new_col'].tolist()))).lower()
        types = ['© single-family','© condo/cooperative','© 2-family', 'o single-family', 'o condo/cooperative','o 2-family']
        t = []
        for i in range(len(types)):
            t.append(re.findall(types[i], s))
        out = []
        [out.extend(s) for s in t]
        if (types[0] in out) and (types[4] in out) and (types[5] in out):
            df_idx[idx+1] = types[0][1:].strip()
        elif (types[1] in out) and (types[3] in out) and (types[5] in out):
            df_idx[idx+1] = types[2][1:].strip()
        elif (types[2] in out) and (types[3] in out) and (types[4] in out):
            df_idx[idx+1] = types[3][1:].strip()
        elif (types[0] in out) and (types[1] in out) and (types[2] in out):
            df_idx[idx+1] = '-'
        else:
            df_idx[idx+1] = '-'
    return df_idx

result = TypeOfBuilding(df_list)
print(f"Dataset Matching Values for key : {keys[4]} \n\n{result}")
# Updating StandardTemplate Dataframe
for k,v in result.items():
    stdf.loc[stdf['FileName'] == k, keys[4]] = v.upper()

### key 6: 'Foundation Information (Flood Zone/BFE/LFE)'

def FoundationInformation(df_list):
    df_idx = {}
    for idx, df in enumerate(df_list, start=0):
        s = str(list(set(df['new_col'].tolist()))).lower()
        types = ['© basement','© slab on grade','© crawl space','© enclosure','obasement','oslab on grade','ocrawl space','oenclosure']
        t = []
        for i in range(len(types)):
            t.append(re.findall(types[i], s))
        out = []
        [out.extend(s) for s in t]
        if (types[0] in out):
            df_idx[idx+1] = types[0][1:].strip()
        elif (types[1] in out):
            df_idx[idx+1] = types[1][1:].strip()
        elif (types[2] in out):
            df_idx[idx+1] = types[2][1:].strip()
        elif (types[3] in out):
            df_idx[idx+1] = types[3][1:].strip()
        else:
            df_idx[idx+1] = '-'
    return df_idx

result = FoundationInformation(df_list)
print(f"Dataset Matching Values for key : {keys[5]} \n\n{result}")
# Updating StandardTemplate Dataframe
for k,v in result.items():
    stdf.loc[stdf['FileName'] == k, keys[5]] = v.upper()

### key 7: 'Homeowner Policy Effective Date'

def homeownerpolicydate(df_list):
    df_idx = {}
    for idx, df in enumerate(df_list, start=0):
        try:
            s = str(set(df_list[idx]['new_col'].tolist()))
            s_idx = re.search('HemeewnerPelicy Effective Date:', s).span()
            v = s[s_idx[1]:s_idx[1]+10].strip()
            df_idx[idx+1] = v
        except:
            continue
    return df_idx

result = homeownerpolicydate(df_list)
print(f"Dataset Matching Values for key : {keys[6]} \n\n{result}")
# Updating StandardTemplate Dataframe
for k,v in result.items():
    stdf.loc[stdf['FileName'] == k, keys[6]] = v.upper()

### key 8: 'Requested Effective Date of Flood Policy'

def RequestedDateofFloodPolicy(df_list):
    df_idx = {}
    for idx, df in enumerate(df_list, start=0):
        try:
            s = str(set(df_list[idx]['new_col'].tolist()))
            s_idx = re.search('Requested Effective Date of Flood Policy:', s).span()
            v = s[s_idx[1]:s_idx[1]+18].strip()
            date = re.findall('[0-9]+', v)
            df_idx[idx+1] = f'{date[0][1]}/{date[1]}/{date[2]}'
        except:
            continue
    return df_idx

result = RequestedDateofFloodPolicy(df_list)
print(f"Dataset Matching Values for key : {keys[7]} \n\n{result}")
# Updating StandardTemplate Dataframe
for k,v in result.items():
    stdf.loc[stdf['FileName'] == k, keys[7]] = v.upper()

### key 9 : 'Building or AA Limits'

def BuildingAALimits(df_list):
    df_idx = {}
    for idx, df in enumerate(df_list, start=0):
        try:
            s = str(set(df_list[idx]['new_col'].tolist())).lower()
            s_idx = re.search('building or a&a ',s).span()
            cost = s[s_idx[1]:s_idx[1]+15]
            c = re.findall('[0-9,]+', cost)
            df_idx[idx+1] = f'${c[0]}'
        except:
            continue
    return df_idx

result = BuildingAALimits(df_list)
print(f"Dataset Matching Values for key : {keys[8]} \n\n{result}")
# Updating StandardTemplate Dataframe
for k,v in result.items():
    stdf.loc[stdf['FileName'] == k, keys[8]] = v.upper()

# Saving submission
stdf.to_csv('C:\\Users\\DharmendraGa_5wskc\\Desktop\\AI Training\\aiLearning\\dharmendra\\feature extraction\\submission.csv', index=False)
print(stdf)

# importing required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# reading the dataset
data = pd.read_csv('final_csv.csv')
#printing top 5 records
data.head()
#adding column of vul_new
data['vul_new'] = data['vulnerabilities'].str.replace("\[|\]", "")
data.head()

#adding column of frequency of vulnerabilities
def add_freq_col_dataset(data):
    col_name = input('enter col_name:')
    freq = {}
    for x in range(len(data[col_name])):
        sg = data[col_name].iloc[x].strip('[]').split(',')

        for x in sg:
            y = x.strip(" ''")
            if y in freq.keys():
                freq[y] += 1
            else:
                freq[y] = 1

    new_col = []
    for x in range(len(data[col_name])):
        sg = data[col_name].iloc[x].strip('[]').split(',')
        sg = [x.strip("'' ") for x in sg]
        cn = []
        for x in sg:
            cn.append(freq[x])
        new_col.append(cn)
    data.insert(len(data.columns), 'new_freq_col', new_col)
    return data.head()

# data.head()
data1 = data.copy()

data1 = data1.dropna(subset=['vul_new'])

#adding freq_column to the dataset
def add_freq_col_dataset(data):
    col_name = input('enter col_name:')
    freq = {}
    for x in range(len(data[col_name])):
        sg = data[col_name].iloc[x].strip('[]').split(',')

        for x in sg:
            y = x.strip(" ''")
            if y in freq.keys():
                freq[y] += 1
            else:
                freq[y] = 1

    new_col = []
    for x in range(len(data[col_name])):
        sg = data[col_name].iloc[x].strip('[]').split(',')
        sg = [x.strip("'' ") for x in sg]
        cn = []
        for x in sg:
            cn.append(freq[x])
        new_col.append(cn)
    data.insert(len(data.columns), 'new_freq_col', new_col)
    return data.head()

#calling the function
add_freq_col_dataset(data1)

data1.drop(['vul_new'], axis=1, inplace=True)

data1[['cpe', 'Type', 'Vendor', 'appname', 'version']] = data1['CPEMatchString'].str.split(':', n=4, expand=True)

data1['Type'].replace(to_replace='/a', value='App', inplace=True)

data1.drop(['cpe'], axis=1, inplace=True)

# last_modified_dates = data1.loc[data1.lastModifiedDate=='lastModifiedDate'].index

# last_modified_dates

# data1.drop(data1.index['254515'],inplace = True)

data1.shape
#saving to csv file
data1.to_csv('sample_data1.csv')
data2 = pd.read_csv('sample_data1.csv')
#printing final dataset
print(data2)
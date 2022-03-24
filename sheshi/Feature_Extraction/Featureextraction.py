import numpy as np
import pandas as pd
import nltk
import re
import os
print(os.getcwd())

data=pd.read_csv("../../mukund/StandardTemplate.csv", encoding='cp1252')
pd.set_option('display.max_colwidth',1000)
pd.set_option('display.max_columns',500)
df0 = pd.read_csv('../../dataset/1.csv')
df1 = pd.read_csv('../../dataset/2.csv')
df2 = pd.read_csv('../../dataset/3.csv')
df3 = pd.read_csv('../../dataset/4.csv')
df4 = pd.read_csv('../../dataset/5.csv')
df5 = pd.read_csv('../../dataset/6.csv')
list_inp=[df0,df1,df2,df3,df4,df5]

def replace_value(regular_exp,column):
    list1=[]
    for i in range(6):
        try:
            data1 = ' '.join(list(list_inp[i]['text'].values))
            match = re.findall(regular_exp, data1)
            if len(match) == 1:
                mail=match[0].split(':')
                list1.append(mail[1])
            else:
                list1.append(np.nan)

        except Exception as e:

            list1.append(np.nan)

    print(list1)


    for i in range(6):
        data.loc[i,column] = list1[i]


regular_exp1 = (r'[A-Za-z]+:\s[a-zA-Z0-9]*[@_]*[a-zA-Z0-9]+@[a-zA-Z]+\.[a-zA-Z]+')
replace_value(regular_exp1,'Email')
reg1=(r'Amount[(s)]* of loss:\$[\d]+\.\d+')
replace_value(reg1,'Amounts of loss')
reg2=(r'Lowest Floor Elevation:\s\w+\s')
replace_value(reg2,'Lowest Floor Elevation')
reg3=(r'Highest Adjacent Grade (HAG):\s\w+\s')
replace_value(reg3,'Highest Adjacent Grade')
reg4=(r'Lowest Adjacent Grade (LAG):\s\w+\s')
replace_value(reg4,'Lowest Adjacent Grade')
data.head()
data.to_csv('submission_featureextraction.csv',index=False)
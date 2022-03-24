import re
import pandas as pd


# Compare below  keys with 6 different dataframes to update StandardTemplate.csv


# Current Flood Policy
# Requested Policy Effective Date
# Writing Company
# Producer Name
# Producer Street Address
# Producer City/State/Zipcode
# Producer Phone Number
# Producer Number


# importing Modules
import warnings
warnings.filterwarnings('ignore')


df4 = pd.read_csv('../../dataset/4.csv', encoding='utf-8')

df4.head()

data1 = ' '.join(list(df4['text'].values))
# data1

df5 = pd.read_csv('../../dataset/5.csv')

# df5

with open('../../dataset/csv5.txt') as f:
    lines = f.readlines()


# lines

df6 = pd.read_csv('../../dataset/6.csv')

# df6

data2 = ' '.join(list(df6['text'].values))
# data2

df3 = pd.read_csv('../../dataset/3.csv')

# df3

data3 = ' '.join(list(df3['text'].values))
# data3

# Requested Policy Effective Date-4,5,6----values also 4,5,6

# csv-4


pattern = r"Requested Policy Effective Date: \d+/+\d+/+\d+ \d+:+\d+ [A-z][A-z]"

u=re.findall(pattern,data1)

# print(u)

#csv-5

pattern = r"Requested Policy Effective Date: \d+/+\d+/+\d+ \d+:+\d+ [A-z][A-z]"

u=re.findall(pattern,str(lines))

# print(u)

# csv-6

pattern = r"Requested Policy Effective Date: \d+/+\d+/+\d+"

u=re.findall(pattern,data2)

# print(u)

# Producer Name -3,4,5,6---- only values have in 3,4,6

# Producer Name RSC Insurance Brokerage

# csv-3

pattern = r"PRODUCER NAME: \w+ \w+ \w+ \w+"
u=re.findall(pattern,data3)
# print(u)

# csv-4

pattern = r"Producer Name \w+ \w+ \w+"
u=re.findall(pattern,data1)
# print(u)

# csv-6

pattern = r"Producer Name _\w+ \w+"
u=re.findall(pattern,data2)
# print(u)

# Writing Company 4,6... values also in 4,6

# csv-4

pattern ="Writing Company: \w+ \w+ \w+"
u=re.findall(pattern,data1)
# print(u)

# csv-6

pattern ="Writing Company: \w+ \w+ \w+"
u=re.findall(pattern,data2)
# print(u)

# Producer Street Address 80 West Century Road, Suite 301

# Producer Street Address-4,5,6...only values 4,6

# csv 4
pattern = r"Producer Street Address \d+ \w+ \w+ \w+,+ \w+ \d+"
u=re.findall(pattern,data1)
# print(u)

# csv 6
pattern = r"Producer Street Address _ \d+ \w+ \w+ \d+"
u=re.findall(pattern,data2)
# print(u)

# Producer City/State/Zipcode - 4,6...values 4,6


# csv:4
pattern = r"Producer City/State/Zipcode \w+,+ \w+ \d+"
u=re.findall(pattern,data1)
# print(u)

#csv:6
pattern = r"Producer City/State/Zipcode __ \w+ \w+,+ \w+ \d+"
u=re.findall(pattern,data2)
# print(u)

# Producer Number-4,6---- values also in 4,6


# csv:4
pattern = r"Producer Number \d+"
u=re.findall(pattern,data1)
# print(u)

# csv:6
pattern = r"Producer Number _ \d+"
u=re.findall(pattern,data2)
# print(u)

# Current Flood Policy -4,5....no values in 4,5,,,,



# Producer Phone Number-4,5,6....values 4,5,6


# cvs:4
pattern = r"Producer Phone Number \d+-+\d+-\d+"
u=re.findall(pattern,data1)
# print(u)

# csv:5
pattern = r"Producer Phone Number \(\d+\) \d+-+\d+"
u=re.findall(pattern,str(lines))
# print(u)

# csv:6
pattern = r"Producer Phone Number_\d+-+\d+-\d+"
u=re.findall(pattern,data2)
# print(u)

# Standard Template Dataset
standard = pd.read_csv('../../dataset/StandardTemplate.csv', encoding='cp1252')
stdf = standard[standard.columns.tolist()[:167]]
# stdf


df = pd.DataFrame(stdf)

df.columns[86:93]

df.iloc[3,86]='91/14/2022 12:01 AM'

df.iloc[4,86]='12/14/2022 12:01 AM'

df.iloc[5,86]='2/1/22'

df.iloc[3,87]='Federal Insurance Company'

df.iloc[5,87]='FEDERAL INSURANCE COMPANY'

df.iloc[3,88]= '_ KRA Insurance Agency'

df.iloc[4,88]='RSC Insurance Brokerage'

df.iloc[5,88]= '_AssuredPartners Northeast'

df.iloc[3,89]='80 West Century Road, Suite 30'

df.iloc[5,89]='_ 445 Hamilton Ave 10'

df.iloc[3,90]='Paramus, NJ 07652'

df.iloc[5,90]='__ White Plains, NY 10601'

df.iloc[3,91]='201-308-8886'

df.iloc[4,91]='(800) 777-2131'

df.iloc[5,91]='_914-761-9000'

df.iloc[3,92]='63336'

df.iloc[5,92]='_ 64709'

df.to_csv('StandardTemplate.csv',encoding='cp1252')


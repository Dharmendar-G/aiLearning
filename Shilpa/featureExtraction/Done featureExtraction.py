import pandas as pd
import re

df1 = pd.read_csv('../../dataset/1.csv',encoding='utf-8')

df4

data = ' '.join(list(df4['text'].values))
data

# Corresponding Chubb Homeowners policy number _14762930-04

pattern = r"Corresponding Chubb Homeowners policy number _\d+"

u=re.findall(pattern,data)

u

# If Contents coverage is being requested, what is the contents amount of coverage on the Chubb Homeowners Policy?'$ 1325.70'

s = str(df4[df4['block_num']==15]['new_col'])

s

s =re.search('\$+ \d+.+',s).group()

s

# If House coverage is being requested, what is the house amount of coverage on the Chubb Homeowners Policy? '$ 4,419,000 '

#below one doubt

#If House coverage is being requested, what is the house amount of coverage on the Chubb Homeowners Policy? $ 4,419,000

s1=df4["new_col"][133:140]

s1

data = ' '.join(list(df4['text'].values))
data

pattern = r"If House coverage is being requested, what is the house amount of coverage on the Chubb Homeowners Policy? \d+"

u=re.findall(pattern,data)

u

#s =re.search('\$+ \d+.+',s).group()

#s2

# City/State/Zipcode:

#City/State/Zipcode: Scarsdale, NY 10583

pattern = r"City/State/Zipcode: \w+, \w+ \d+"

u=re.findall(pattern,data)

print(u)

df6 = pd.read_csv('../../dataset/6.csv',encoding='utf-8')

df6

data = ' '.join(list(df6['text'].values))
data

# Corresponding Chubb Homeowners policy number _ 1523893801 __

pattern = r"Corresponding Chubb Homeowners policy number _\d+_"

u=re.findall(pattern,data)


u

# If Contents coverage is being requested, what is the contents amount of coverage on the Chubb Homeowners Policy?' $2,070,000.'

s3 = str(df6[df6['block_num']==14]['new_col'])

s3

#s3 =re.search('\$+ \d+.+',s3).group()

s3

# If House coverage is being requested, what is the house amount of coverage on the Chubb Homeowners Policy? '$6,900,000 '

s4 = str(df6[df6['block_num']==13]['new_col'])

s4

s5= re.search('$ +\d+.',s4)

s5

# City/State/Zipcode:

#City/State/Zipcode: Lido Beach, NY 11561

pattern = r"City/State/Zipcode: \w+, \w+ \d+"

u=re.findall(pattern,data)

print(u)

# Standard Template Dataset
standard = pd.read_csv('StandardTemplate.csv', encoding='cp1252')
stdf = standard[standard.columns.tolist()[:167]]

stdf

df = pd.DataFrame(stdf)

df.columns[99:103]

df.iloc[4,99]= 'Scarsdale, NY 10583'

df.iloc[4,100]= 'Corresponding Chubb Homeowners policy number _14762930'

df.iloc[4,101]= 'If Contents coverage is being requested, what is the contents amount of coverage on the Chubb Homeowners Policy?$ 1325.70'

df.iloc[4,102]= 'If House coverage is being requested, what is the house amount of coverage on the Chubb Homeowners Policy?$ 4,419,000'
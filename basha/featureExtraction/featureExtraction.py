import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import re

#Read the csv files
df1 = pd.read_csv('../../dataset/1.csv',encoding='utf-8')
df2 = pd.read_csv('../../dataset/2.csv',encoding='utf-8')
df3 = pd.read_csv('../../dataset/3.csv',encoding='utf-8')
df4 = pd.read_csv('../../dataset/4.csv',encoding='utf-8')
with open('../../dataset/5csv_complete_text.txt') as f:
    df5 = f.readlines()
df6 = pd.read_csv('../../dataset/6.csv')

#Extract the text from csv and join the lines
data1 = ' '.join(list(df1['text'].values))
data2 = ' '.join(list(df2['text'].values))
data3= ' '.join(list(df3['text'].values))
data4= ' '.join(list(df4['text'].values))
data6= ' '.join(list(df6['text'].values))

# check Flood Zone feature present in each csv file or not
result1_1 = []
find = ['Flood Zone:']
for i in find:
    match_string = df1.new_col.str.findall(i)
    match_string = [ele for ele in match_string if ele != [] and ele != 'NaN' and ele != 'nan']
    result1_1.append(match_string)

result1_2 = []
find = ['Flood Zone:']
for i in find:
    match_string = df2.new_col.str.findall(i)
    match_string = [ele for ele in match_string if ele != [] and ele != 'NaN' and ele != 'nan']
    result1_2.append(match_string)

result1_3 = []
find = ['Flood Zone:']
for i in find:
    match_string = df3.new_col.str.findall(i)
    match_string = [ele for ele in match_string if ele != [] and ele != 'NaN' and ele != 'nan']
    result1_3.append(match_string)

result1_4 = []
find = ['Flood Zone:']
for i in find:
    match_string = df4.new_col.str.findall(i)
    match_string = [ele for ele in match_string if ele != [] and ele != 'NaN' and ele != 'nan']
    result1_4.append(match_string)

result1_6 = []
find = ['Flood Zone:']
for i in find:
    match_string = df6.new_col.str.findall(i)
    match_string = [ele for ele in match_string if ele != [] and ele != 'NaN' and ele != 'nan']
    result1_6.append(match_string)

# check 'Month & Year Built/Substantial Improvement Date'feature present in csv file or not
result2_1 = []
find = ['Month & Year Built/Substantial Improvement Date']
for i in find:
    match_string = df1.new_col.str.findall(i)
    match_string = [ele for ele in match_string if ele != [] and ele != 'NaN' and ele != 'nan']
    result2_1.append(match_string)

result2_2 = []
find = ['Month & Year Built/Substantial Improvement Date']
for i in find:
    match_string = df2.new_col.str.findall(i)
    match_string = [ele for ele in match_string if ele != [] and ele != 'NaN' and ele != 'nan']
    result2_2.append(match_string)

result2_3 = []
find = ['Month & Year Built/Substantial Improvement Date']
for i in find:
    match_string = df3.new_col.str.findall(i)
    match_string = [ele for ele in match_string if ele != [] and ele != 'NaN' and ele != 'nan']
    result2_3.append(match_string)

result2_4 = []
find = ['Month & Year Built/Substantial Improvement Date']
for i in find:
    match_string = df4.new_col.str.findall(i)
    match_string = [ele for ele in match_string if ele != [] and ele != 'NaN' and ele != 'nan']
    result2_4.append(match_string)
result2_4.append(match_string)

result2_6 = []
find = ['Month & Year Built/Substantial Improvement Date']
for i in find:
    match_string = df6.new_col.str.findall(i)
    match_string = [ele for ele in match_string if ele != [] and ele != 'NaN' and ele != 'nan']
    result2_6.append(match_string)

# check 'Month & Year Built/Substantial Improvement Date'feature present in csv file or not
result3_1 = []
find = ['County/Parish']
for i in find:
    match_string = df1.new_col.str.findall(i)
    match_string = [ele for ele in match_string if ele != [] and ele != 'NaN' and ele != 'nan']
    result3_1.append(match_string)

result3_2 = []
find = ['County/Parish']
for i in find:
    match_string = df2.new_col.str.findall(i)
    match_string = [ele for ele in match_string if ele != [] and ele != 'NaN' and ele != 'nan']
    result3_2.append(match_string)

result3_3 = []
find = ['County/Parish']
for i in find:
    match_string = df3.new_col.str.findall(i)
    match_string = [ele for ele in match_string if ele != [] and ele != 'NaN' and ele != 'nan']
    result3_3.append(match_string)

result3_4 = []
find = ['County/Parish']
for i in find:
    match_string = df4.new_col.str.findall(i)
    match_string = [ele for ele in match_string if ele != [] and ele != 'NaN' and ele != 'nan']
    result3_4.append(match_string)

result3_6 = []
find = ['County/Parish']
for i in find:
    match_string = df6.new_col.str.findall(i)
    match_string = [ele for ele in match_string if ele != [] and ele != 'NaN' and ele != 'nan']
    result3_6.append(match_string)

# check'If Other permanent structures coverage is requested, what is the other permanent structures total value?'in all csvs
result4_1 = []
find = ['If Other permanent structures coverage is requested, what is the other permanent structures total value?']
for i in find:
    match_string = df1.new_col.str.findall(i)
    match_string = [ele for ele in match_string if ele != [] and ele != 'NaN' and ele != 'nan']
    result4_1.append(match_string)

result4_2 = []
find = ['If Other permanent structures coverage is requested, what is the other permanent structures total value?']
for i in find:
    match_string = df2.new_col.str.findall(i)
    match_string = [ele for ele in match_string if ele != [] and ele != 'NaN' and ele != 'nan']
    result4_2.append(match_string)

result4_3 = []
find = ['If Other permanent structures coverage is requested, what is the other permanent structures total value?']
for i in find:
    match_string = df3.new_col.str.findall(i)
    match_string = [ele for ele in match_string if ele != [] and ele != 'NaN' and ele != 'nan']
    result4_3.append(match_string)

result4_4 = []
find = ['If Other permanent structures coverage is requested, what is the other permanent structures total value?']
for i in find:
    match_string = df4.new_col.str.findall(i)
    match_string = [ele for ele in match_string if ele != [] and ele != 'NaN' and ele != 'nan']
    result4_4.append(match_string)

result4_6 = []
find = ['If Other permanent structures coverage is requested, what is the other permanent structures total value?']
for i in find:
    match_string = df6.new_col.str.findall(i)
    match_string = [ele for ele in match_string if ele != [] and ele != 'NaN' and ele != 'nan']
    result4_6.append(match_string)

# Based on above checking keys we have both key and values in csv_6 only..i have considerd csv 6
pattern = r"Month & Year Built/Substantial Improvement Date: \w/[0-9]+"
u=re.findall(pattern,data6)

pattern = r"County/Parish: \w+ \w+ \w+"
u=re.findall(pattern,data6)

pattern = r"Flood Zone: __[A-z]"
u=re.findall(pattern,data6)

#If Other permanent structures coverage is requested, what is the other permanent structures total value?'
s = str(df6[df6['block_num']==15]['new_col'])
s =re.search('\$\d+,\d+,\d+', s).group()

# Standard Template Dataset
standard = pd.read_csv('../../dataset/StandardTemplate.csv', encoding='cp1252')
stdf = standard[standard.columns.tolist()[:167]]
stdf.insert(0, column='FileName', value=[x for x in range(1,7)])
stdf
df = pd.DataFrame(stdf)
df.columns[101:105]
df.iloc[5,101]= '$1,380,000'
df.iloc[5,102]= 'X'
df.iloc[5,103]= '01-2011'
df.iloc[5,104]= 'Palm Beach County'
df.to_csv('submission.csv',encoding='cp1252')
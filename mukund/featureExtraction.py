import os
import pandas as pd
path = os.getcwd()
files = []
for i in range(1,7):
    for file in os.listdir():
        if file.endswith(f"{i}.csv"):
            files.append(file)
print(files)

dfs = [pd.read_csv(f, low_memory=False).drop('Unnamed: 0',axis=1) for f in files]

li_st=[]
for x in dfs:
    li_st.append(list(x.text))
print("Length of list of dataframes that contains only text columns",len(li_st))

list_of_df_str=[]
for text in li_st:
    list_of_df_str.append(" ".join(map(str,text)))
print(len(list_of_df_str))
# containing list of every dataframe text string
csv1_text=list_of_df_str[0]
print('*********************************')
print('Text of csv file 1---------------',csv1_text)
print('\n*********************************\n')
csv2_text=list_of_df_str[1]
print('Text of csv file 2-----------------',csv2_text)
print('\n*********************************\n')
csv3_text=list_of_df_str[2]
print('Text of csv file 3-----------------',csv3_text)
print('\n*********************************\n')
csv4_text=list_of_df_str[3]
print('Text of csv file 4-----------------',csv4_text)
print('\n*********************************\n')
csv5_text=list_of_df_str[4]
print('Text of csv file 5-----------',csv5_text)
print('\n*********************************\n')
csv6_text=list_of_df_str[5]
print('Text of csv file 6 ---------',csv6_text)
print('\n*********************************\n')
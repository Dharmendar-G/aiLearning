import os,re
import pandas as pd
df=pd.read_csv('StandardTemplate.csv',encoding='cp1252')
col=list(df.columns)

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


def flood_zone(df_str):
    while re.search("Flood Zone",df_str):
        num=re.search("Flood Zone",df_str).end()
        fl_zone=df_str[num:num+6]
        zone=re.search("[x|X]",fl_zone)
        if zone:
            return zone.group()
        break


def bfe(df_str):
    while re.search("BFE:",df_str):
        num=re.search("BFE:",df_str).end()
        bfe_1=df_str[num:num+5]
        while ' ' in bfe_1:
            bfe_1 = bfe_1.replace(' ', '')
        if bfe_1=="____":
            return None
        break

def lfe(df_str):
    while re.search("LFE:",df_str):
        num=re.search("LFE:",df_str).end()
        lfe_1=df_str[num:num+3]
        while ' ' in lfe_1:
            lfe_1 = lfe_1.replace(' ', '')
        if lfe_1=="FL":
            return None
        break

def ALE(df_str):
    while re.search("additional living expenses",df_str.lower()):
        num=re.search("additional living expenses",df_str.lower()).end()
        ale_1=df_str[num+2:num+9]
        while ' ' in ale_1:
            ale_1 = ale_1.replace(' ', '')
        if re.search("[\d]",ale_1):
            ale_1=ale_1[:8]
            return ale_1
        return None

def Quote_form(df_str):
    while re.search("Date Quote Form Submitted to WNC:",df_str):
        num=re.search("Date Quote Form Submitted to WNC:",df_str).end()
        date=df_str[num+1:num+11]
        if date[0]=='9':
            date=date.replace(date[0],'0')
            if re.search("[a-zA-Z]",date):
                date=date[:re.search("[a-zA-Z]",date).start()]
                return date
            return date
        break
    else:
        return None


def building(df_str):
    while re.search("Building or A",df_str):
        num=re.search("Building or A",df_str).end()
        bui_ld=df_str[num+4:num+13]
        break
    else:
        return None

def deduct(df_str):
    if re.search("Deductible ",df_str):
        num=re.search("Deductible ",df_str).end()
        var=df_str[num+13:num+20]
        if re.search("[a-zA-Z]",var):
            var=var[:re.search("[a-zA-Z]",var).start()]
        return var

def contents(df_str):
    while re.search("Contents ",df_str):
        num=re.search("Contents ",df_str).end()
        var=df_str[num:num+15]
        if re.search("[1-9]",var):
            var=var[re.search("[1-9]",var).start():re.search("[a-zA-Z]",var).start()]
            return var
        break

def con_limit(df_str):
    while re.search("content limit",df_str.lower()):
        return None


to_csv1={"Flood Zone":flood_zone(csv1_text),"BFE":bfe(csv1_text),"LFE":lfe(csv1_text),
         "Additional Living Expenses ALE":ALE(csv1_text),"Date Quote Form Submitted to WNC":Quote_form(csv1_text),
        "Contents":contents(csv1_text),"Building or AA":building(csv1_text),"Contents Limit":con_limit(csv1_text),"Deductible":deduct(csv1_text)}

to_csv2={"Flood Zone":flood_zone(csv2_text),"BFE":bfe(csv2_text),"LFE":lfe(csv2_text),
         "Additional Living Expenses ALE":ALE(csv2_text),"Date Quote Form Submitted to WNC":Quote_form(csv2_text),
        "Contents":contents(csv2_text),"Building or AA":building(csv2_text),"Contents Limit":con_limit(csv2_text),"Deductible":deduct(csv2_text)}

to_csv3={"Flood Zone":flood_zone(csv3_text),"BFE":bfe(csv3_text),"LFE":lfe(csv3_text),
         "Additional Living Expenses ALE":ALE(csv3_text),"Date Quote Form Submitted to WNC":Quote_form(csv3_text),
        "Contents":contents(csv3_text),"Building or AA":building(csv3_text),"Contents Limit":con_limit(csv3_text),"Deductible":deduct(csv3_text)}

to_csv4={"Flood Zone":flood_zone(csv4_text),"BFE":bfe(csv4_text),"LFE":lfe(csv4_text),
         "Additional Living Expenses ALE":ALE(csv4_text),"Date Quote Form Submitted to WNC":Quote_form(csv4_text),
        "Contents":contents(csv4_text),"Building or AA":building(csv4_text),"Contents Limit":con_limit(csv4_text),"Deductible":deduct(csv4_text)}

to_csv5={"Flood Zone":flood_zone(csv5_text),"BFE":bfe(csv5_text),"LFE":lfe(csv5_text),
         "Additional Living Expenses ALE":ALE(csv5_text),"Date Quote Form Submitted to WNC":Quote_form(csv5_text),
        "Contents":contents(csv5_text),"Building or AA":building(csv5_text),"Contents Limit":con_limit(csv5_text),"Deductible":deduct(csv5_text)}

to_csv6={"Flood Zone":flood_zone(csv6_text),"BFE":bfe(csv6_text),"LFE":lfe(csv6_text),
         "Additional Living Expenses ALE":ALE(csv6_text),"Date Quote Form Submitted to WNC":Quote_form(csv6_text),
        "Contents":contents(csv6_text),"Building or AA":building(csv6_text),"Contents Limit":con_limit(csv6_text),"Deductible":deduct(csv6_text)}



from csv import DictWriter

def append_dict_as_row(file_name, dict_of_elem, field_names):
    with open(file_name, 'a+', newline='') as obj:
        # Create a writer object from csv module
        dict_writer = DictWriter(obj, fieldnames=field_names)
        # Add dictionary as word in the csv
        dict_writer.writerow(dict_of_elem)


append_dict_as_row('StandardTemplate.csv', to_csv1, col)  # for col .. go to line no 3
append_dict_as_row('StandardTemplate.csv', to_csv2, col)
append_dict_as_row('StandardTemplate.csv', to_csv3, col)
append_dict_as_row('StandardTemplate.csv', to_csv4, col)
append_dict_as_row('StandardTemplate.csv', to_csv5, col)
append_dict_as_row('StandardTemplate.csv', to_csv6, col)

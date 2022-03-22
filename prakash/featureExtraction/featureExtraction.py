
print('hello')
# Problem Statement
# Compare below 5 keys with 6 different dataframes to update StandardTemplate.csv and inserting in it(StandardTemplate.csv).

# Contents in Basement

# Real Property in Basement

# Rebuilding to Code

# CHUBB PERSONAL (PRIMARY) FLOOD OR
# CHUBB EXCESS FLOOD

# PRODUCER NAME

# 1)Contents in Basement: present in 2.csv file only  and the  value is represented as maximum.

# 2)Real Property in Basement : present in 2.csv file only and the value is represented as maximum.

# 3)Rebuilding to Code  :present in 2.csv file only and the value is represented as maxium.

# 4)CHUBB PERSONAL (PRIMARY) FLOOD OR
# CHUBB EXCESS FLOOD                  : not found in any csv file.


# 5)PRODUCER NAME      : present in 3.csv file only and the value is represented as  _ KRA Insurance Agency

#importing Required Libraries
import pandas as pd
#import numpy as np
#import re
import nltk
import warnings
warnings.filterwarnings("ignore")
#reading the all datasets
data1 = pd.read_csv('1.csv')
data2 = pd.read_csv('2.csv')
data3 = pd.read_csv('3.csv')
data4 = pd.read_csv('2.csv')
data5 = pd.read_csv('5.csv')
data6 = pd.read_csv('6.csv')
stan_data = pd.read_csv('StandardTemplate.csv',encoding='cp1252')
#stan_data.head()
#dropping the null values for the all  datasets from 1 to 6
data1 = data1.dropna()
data2 = data2.dropna()
data3 = data3.dropna()
data4 = data4.dropna()
data5 = data5.dropna()
data6 = data6.dropna()

# Contents in Basement key

#reading the given key as string
str1 = 'Contents in Basement'
# writing function to retrieve the values of new_col from every dataset
output_final = []
def output_list(str):
    for i in range(1, 7):
        data = pd.read_csv(f"{i}.csv")
        data = data.dropna()
        csv = data[data['new_col'].str.contains(str1)]
        a = csv['new_col'].unique()
        lis1 = list(a)
        # print(lis1)
        lis2 = (''.join(lis1))

        string = lis2
        sub_str = 'Maximum'
        if (string.find(sub_str) == -1):
            print('for  data{} value not found'.format(i))
            output_final.append('')
            # output_lis2.append(output_lis1)
            # return output_final
        else:
            print('for  data{} value found'.format(i))

            def listtostr(lis1):
                str3 = " "
                for e in lis1:
                    str3 += e
                return str3

            str5 = listtostr(lis1)
            print(str5)

            def lastword(str):
                newstr = ''
                length = len(str)
                for i in range(length - 1, 0, -1):
                    if (str[i] == " "):
                        return newstr[::-1]
                    else:
                        newstr = newstr + str[i]

            op2 = lastword(str5)
            print(op2)
            output_final.append(op2)

    return output_final

#calling function
output_list(str1)

#Real Property in Basement key

#reading the given key as string
str2 = 'Real Property in Basement'
# writing function to retrieve the values of new_col from every dataset
output_final2 = []
def output_list2(str):
    for i in range(1, 7):
        data = pd.read_csv(f"{i}.csv")
        data = data.dropna()
        csv = data[data['new_col'].str.contains(str2)]
        a = csv['new_col'].unique()
        lis1 = list(a)
        # print(lis1)
        lis2 = (''.join(lis1))

        string = lis2
        sub_str = 'Maximum'
        if (string.find(sub_str) == -1):
            print('for  data{} value not found'.format(i))
            output_final2.append('')
            # output_lis2.append(output_lis1)
            # return output_final
        else:
            print('for  data{} value found'.format(i))

            def listtostr(lis1):
                str3 = " "
                for e in lis1:
                    str3 += e
                return str3

            str5 = listtostr(lis1)
            print(str5)

            def lastword(str):
                newstr = ''
                length = len(str)
                for i in range(length - 1, 0, -1):
                    if (str[i] == " "):
                        return newstr[::-1]
                    else:
                        newstr = newstr + str[i]

            op2 = lastword(str5)
            print(op2)
            output_final2.append(op2)

    return output_final2

#calling function
output_list2(str2)

#Rebuilding to Code key

#representing the given key as string
str3 = 'Rebuilding to Code'
##function for retrieve values of new_col from all datasets
output_final3 = []


def output_list3(str):
    for i in range(1, 7):
        data = pd.read_csv(f"{i}.csv")
        data = data.dropna()
        csv = data[data['new_col'].str.contains(str3)]
        a = csv['new_col'].unique()
        lis1 = list(a)
        # print(lis1)
        lis2 = (''.join(lis1))
        # print(lis2)

        string = lis2
        sub_str = 'Maxium'
        if (string.find(sub_str) == -1):
            print('for  data{} value not found'.format(i))
            output_final3.append('')
            # output_lis2.append(output_lis1)
            # return output_final
        else:
            print('for  data{} value found'.format(i))

            def listtostr(lis1):
                str3 = " "
                for e in lis1:
                    str3 += e
                return str3

            str5 = listtostr(lis1)
            print(str5)

            def lastword(str):
                newstr = ''
                length = len(str)
                for i in range(length - 1, 0, -1):
                    if (str[i] == " "):
                        return newstr[::-1]
                    else:
                        newstr = newstr + str[i]

            op2 = lastword(str5)
            print(op2)
            output_final3.append('Maxium')

    return output_final3

#calling function
output_list3(str3)

#CHUBB PERSONAL (PRIMARY) FLOOD OR CHUBB EXCESS FLOOD key

#representing the given key as string
str4 = 'FLOOD OR CHUBB EXCESS FLOOD'
##function for retrieve values of new_col from all datasets
output_final4 = []
def output_list4(str):
    for i in range(1, 7):
        data = pd.read_csv(f"{i}.csv")
        data = data.dropna()
        csv = data[data['new_col'].str.contains(str4)]
        a = csv['new_col'].unique()
        lis1 = list(a)
        # print(lis1)
        lis2 = (''.join(lis1))
        # print(lis2)

        string = lis2
        sub_str = '____'
        if (string.find(sub_str) == -1):
            print('for  data{} value not found'.format(i))
            output_final4.append('')
            # output_lis2.append(output_lis1)
            # return output_final
        else:
            print('for  data{} value found'.format(i))

            def listtostr(lis1):
                str3 = " "
                for e in lis1:
                    str3 += e
                return str3

            str5 = listtostr(lis1)
            print(str5)

            def lastword(str):
                newstr = ''
                length = len(str)
                for i in range(length - 1, 0, -1):
                    if (str[i] == " "):
                        return newstr[::-1]
                    else:
                        newstr = newstr + str[i]

            op2 = lastword(str5)
            print(op2)
            output_final4.append(op2)

    return output_final4

#calling function
output_list4(str4)

#PRODUCER NAME key

#representing the given key as string
str55 = 'PRODUCER NAME'
output_final5 = []
def output_list5(str):
    for i in range(1, 7):
        data = pd.read_csv(f"{i}.csv")
        data = data.dropna()
        csv = data[data['new_col'].str.contains(str55)]
        a = csv['new_col'].unique()
        lis1 = list(a)
        # print(lis1)
        lis2 = (''.join(lis1))
        # print(lis2)

        string = lis2
        sub_str = '_ KRA Insurance Agency'
        if (string.find(sub_str) == -1):
            print('for  data{} value not found'.format(i))
            output_final5.append('')
            # output_lis2.append(output_lis1)
            # return output_final
        else:
            print('for  data{} value found'.format(i))

            def listtostr(lis1):
                str3 = " "
                for e in lis1:
                    str3 += e
                return str3

            str5 = listtostr(lis1)
            print(str5)

            def value_str(str):
                x = str.split(':')
                value = x[1][:23]
                return value

            op2 = value_str(str5)
            print(op2)
            output_final5.append(op2)

    return output_final5

#calling function
output_list5(str55)

#list of output_lists and columns
out_lists = [output_final,output_final2,output_final3,output_final4,output_final5]
col_names = ['Contents in Basement1','Real Property in Basement1','Rebuilding to Code1','CHUBB PERSONAL (PRIMARY) FLOOD OR CHUBB EXCESS FLOOD1','PRODUCER NAME1']

#inserting new columns into standard dataset
for i in range(5):
    stan_data[col_names[i]] = out_lists[i]

#converting standard datafile  to csv
stan_data.to_csv('updated_standard_template.csv',index=False)

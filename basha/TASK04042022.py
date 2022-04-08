import pandas as pd
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv('samplefile.csv',encoding='utf-8')

# using constructor

# Action
class action:
    def __init__(self, data):
        self.df = data

    def action_updated(self):
        result = []
        find = ['action:\w+']
        for i in find:
            ful = self.df.new_col.str.findall(i)
            ful = [ele for ele in ful if ele != [] and ele != 'NaN' and ele != 'nan']
            result.append(ful)
        # print(result)


act1 = action(df)
act1.action_updated()


# Epochs
class action:
    def __init__(self, data):
        self.df = data

    def action_updated(self):
        result = []
        find = ['\w+:[0-9]+']
        for i in find:
            ful = self.df.new_col.str.findall(i)
            ful = [ele for ele in ful if ele != [] and ele != 'NaN' and ele != 'nan']
            result.append(ful)
        # print(result)


act1 = action(df)
act1.action_updated()


# Package
class action:
    def __init__(self, data):
        self.df = data

    def action_updated(self):
        result = []
        find = ['\w+:\w+\w+-[0-9]+.[0-9]+.[0-9]+.[0-9].\w+.\w+']
        for i in find:
            ful = self.df.new_col.str.findall(i)
            ful = [ele for ele in ful if ele != [] and ele != 'NaN' and ele != 'nan']
            result.append(ful)
        # print(result)


act1 = action(df)
act1.action_updated()
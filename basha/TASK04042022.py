import pandas as pd
import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('2.csv')
data_twenty = data.head(30)
data_twenty.drop('Unnamed: 0', inplace=True, axis=1)
data_twenty['new_col'][1]
data_twenty['new_col'][2]
data_twenty.loc[::2, 'new_col'] = 'CHUBB QUICK FLOOD ESTIMATE WORKSHEET (@) PERSONAL (PRIMARY) FLOOD action:Updated'
data_twenty.loc[::3, 'new_col'] = 'Epoch:40'
data_twenty.loc[::2,
'new_col'] = 'CHUBB QUICK FLOOD ESTIMATE WORKSHEET (@) PERSONAL (PRIMARY) FLOOD action:Updated,Package:libcacard-2.7.0-1.el7.x86_64'
data_twenty['new_col'][9]
df = pd.DataFrame(data_twenty)
df.to_csv('Samplefile', encoding='cp1252')


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
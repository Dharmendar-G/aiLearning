# PROBLEM STATEMENT
# Objective:
# The classification goal is to predict the likelihood of a liability customer buying personal loans.
# Steps and tasks:
# 1.            Read the column description and ensure you understand each attribute well
# 2.            Study the data distribution in each attribute, share your findings
# 3.            Get the target column distribution.
# 4.            Split the data into training and test set in the ratio of 70:30 respectively
# 5.            Use different classification models (Logistic, K-NN and Naïve Bayes) to predict the likelihood of a customer buying personal loans
# 6.            Print the confusion matrix for all the above models
# 7.            Give your reasoning on which is the best model in this case and why it performs better?

# LOAD THE DATASET
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('Bank_Personal_Loan_Modelling.csv')
df

# column_Descriptions
# Age:customers age
# Experience: Number of years Experience
# income :Year income of the customer
# Zipcode :Address
# Family:Total size of customer family members
# ccAvg: credit card Average
# Educationlevels 1:undergraduate 2:Graduate 3:Higher professional
# Mortgage:value of the house mortgage(borrows money to buy)
# personal loan : Does customer buying personal loan or not?
# securities amount :customer having a security account with bank or not?
# CD account:customer having a certificate of Deposit account with bank or not?
# online :customer using internet banking facilities or not?
# credit card : customer  using credit card or not?

#check columns
df.columns

#checking the dtypes
df.dtypes

#checking describe
df.describe().T

# check shape
df.shape

#check null values
df.isnull().sum()

#In above we have observed experience having negative values lets check customers having less than zero  experience or not?
df[df['Experience']<0].count()

df[df['Experience']<0]['Experience'].value_counts()

#Drop Id and experience and zipcode it will not effect our modelling
df.drop(['ID','Experience','ZIP Code'],axis=1,inplace=True)

df.head()
df.tail()


#checking unique values
counts = df.nunique()
counts

#Now checking with  no of customers  having zero mortage,zero personal loan ,zero securities account ,Cd account ,credit card,education,online
df[df['Mortgage']==0]['Mortgage'].value_counts()
df[df['CCAvg']==0]['CCAvg'].value_counts()

df['Family'].value_counts()
df['Securities Account'].value_counts()
df['CD Account'].value_counts()
df['Online'].value_counts()
df['CreditCard'].value_counts()
df['Education'].value_counts()

#visulizations
def distplot_age(func):
    def distplot():
        plt.figure(figsize=(15,6))
        plt.subplot(1,2,1)
        print(sns.distplot(df['Age']))
    return distplot()

def distplot_income(func):
    def distplot():
        plt.figure(figsize=(15,6))
        plt.subplot(1,2,1)
        print(sns.distplot(df["Income"]))
    return distplot()

def distplot_Mortgage(func):
    def distplot():
        plt.figure(figsize=(15,6))
        plt.subplot(1,2,1)
        print(sns.distplot(df["Mortgage"]))
    return distplot()

def distplot_CCAVG(func):
    def distplot():
        plt.figure(figsize=(15,6))
        plt.subplot(1,2,1)
        print(sns.distplot(df["CCAvg"]))
    return distplot()

def countplot_finally(func):
    def countplot():
        plt.figure(figsize=(20,6))
        plt.subplot(1,2,1)
        print(sns.countplot(x='Family',data=df))
    return countplot()

def countplot_education(func):
    def countplot():
        plt.figure(figsize=(20,6))
        plt.subplot(1,2,1)
        print(sns.countplot(x='Education',data=df))
    return countplot()

def countplot_cc(func):
    def countplot():
        plt.figure(figsize=(20,6))
        plt.subplot(1,2,1)
        print(sns.countplot(x='CreditCard',data=df))
    return countplot()

def countplot_cc(func):
    def countplot():
        plt.figure(figsize=(20,6))
        plt.subplot(1,2,1)
        print(sns.countplot(x='CreditCard',data=df))
    return countplot()

def countplot_online(func):
    def countplot():
        plt.figure(figsize=(20,6))
        plt.subplot(1,2,1)
        print(sns.countplot(x='Online',data=df))
    return countplot()


@countplot_online
@countplot_cc
@countplot_education
@countplot_finally
@distplot_CCAVG
@distplot_Mortgage
@distplot_income
@distplot_age
def displot():
    print("univarite plots")

def countplot_education(func):
    def countplot():
        plt.figure(figsize=(20,6))
        plt.subplot(1,2,1)
        print(sns.countplot(data=df,x='Education',hue='Personal Loan',palette='RdBu_r'))
    return countplot()

def bar__plot_ed(func):
    def bar__plot():
        plt.figure(figsize=(20,6))
        plt.subplot(1,2,1)
        print(sns.barplot('Education','Mortgage',hue='Personal Loan',data=df,palette='viridis',ci=None))
    return bar__plot()

def countplot_securities(func):
    def countplot():
        plt.figure(figsize=(20,6))
        plt.subplot(1,2,1)
        print(sns.countplot(data=df,x='Securities Account',hue='Personal Loan',palette='Set2'))
    return countplot()

def box__plot_ccavg(func):
    def box__plot():
        plt.figure(figsize=(20,6))
        plt.subplot(1,2,1)
        print(sns.boxplot('CreditCard','CCAvg',hue='Personal Loan',data=df,palette='RdBu_r'))
    return box__plot()

@box__plot_ccavg
@countplot_securities
@bar__plot_ed
@countplot_education
def displot():
    print("bivariate")

#heatmap
plt.subplots(figsize=(12,10))
sns.heatmap(df.corr(),annot = True)

sns.pairplot(df)

#outliers
# Outlier treatment:
def outlier_treatment(data_column):
    sorted(data_column)
    q1,q3 = np.percentile(data_column,[25,75])
    iqr = q3 - q1
    #print(iqr)
    lr = q1 - (1.5 * iqr)
    ur = q3 + (1.5 * iqr)
    return lr,ur

list1 = ['Age', 'Income', 'Family', 'CCAvg', 'Education', 'Mortgage',
       'Personal Loan', 'Securities Account', 'CD Account', 'Online',
       'CreditCard']
for val in list1:
    l1,u1 = outlier_treatment(df[val])
    print(l1,u1)
    df[val] = np.where(df[val] > u1, u1,df[val])
    df[val] = np.where(df[val] < l1, l1,df[val])


plt.figure(figsize=(15,8))
sns.boxplot(data=df)

#check skew whether data normally distrubuted or not.¶
from scipy.stats import skew
for i in df.columns:
    print(skew(df[i],axis=0),'for',i)

def dist_plot(data_column):
    plt.figure(figsize=(15,10))
    sns.distplot(df[data_column], kde = True, color ='blue')
    plt.show()

for val in df.columns:
    dist_plot(val)

X= df.drop(['Personal Loan'],axis=1)
y= df['Personal Loan']


#using  power transformation (feature income and cc avg make them as symmetric because high left skew and right skew)
pt = PowerTransformer(method='yeo-johnson',standardize=False)
pt.fit(X['Income'].values.reshape(-1,1))
pt.fit(X['CCAvg'].values.reshape(-1,1))
income= pt.transform(X['Income'].values.reshape(-1,1))
ccavg = pt.transform(X['CCAvg'].values.reshape(-1,1))
X['Income'] = pd.Series(income.flatten())
X['CCAvg']= pd.Series(ccavg.flatten())

plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
sns.distplot(X['Income'])
plt.subplot(1,2,2)
sns.distplot(X['CCAvg'])
plt.show()

#Splitting of data into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)


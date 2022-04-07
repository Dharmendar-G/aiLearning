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
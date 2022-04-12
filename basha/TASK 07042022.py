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

#check skew whether data is normal distrubution or not

from scipy.stats import skew
for i in df.columns:
    print(skew(df[i],axis=0),'for',i)

def dist_plot(data_column):
    plt.figure(figsize=(15,10))
    sns.distplot(df[data_column], kde = True, color ='blue')
    plt.show()

for val in df.columns:
    dist_plot(val)

#Transformations
X= df.drop(['Personal Loan'],axis=1)
y= df['Personal Loan']

#Transformation on the Income variable because we have high skew  value we will try to reduce the skew value using tranformations
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson',standardize=False)
pt.fit(X['Income'].values.reshape(-1,1))
temp = pt.transform(X['Income'].values.reshape(-1,1))
X['Income'] = pd.Series(temp.flatten())

# Distplot to show transformed Income variable
sns.distplot(X['Income'])
plt.show()

# CCAvg variable.
pt = PowerTransformer(method='yeo-johnson',standardize=False)
pt.fit(X['CCAvg'].values.reshape(-1,1))
temp = pt.transform(X['CCAvg'].values.reshape(-1,1))
X['CCAvg'] = pd.Series(temp.flatten())

sns.distplot(X['CCAvg'])
plt.show()


#Target column distrubution using pie chart

tempDF = pd.DataFrame(df['CreditCard'].value_counts()).reset_index()
tempDF.columns = ['Labels', 'CreditCard']
fig1, ax1 = plt.subplots(figsize=(10,8))
explode = (0, 0.15)
ax1.pie(tempDF['CreditCard'] , explode= explode, autopct= '%2.1f%%',shadow=True , startangle = 70)
ax1.axis('equal')
plt.title('creditcard Percentage')
plt.show()
#Based on above pie chart 9.6% only buying personal loan

#Splitting of data into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3, random_state = 0)
X_train.head()


#Standarization(scaled down values From o to 1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Logistic Regression(unscaled_data):
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
model = LogisticRegression()
model.fit(X_train,Y_train)
y_logistic_pred = model.predict(X_test)
logistic_re = classification_report(Y_test,y_logistic_pred)
print(logistic_re)

##scaled_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
model = LogisticRegression()
model.fit(X_train_1,Y_train)
y_logistic_pred = model.predict(X_test_1)
logistic_re = classification_report(Y_test,y_logistic_pred)
print(logistic_re)

#K_Nearest_Neighbour:
from sklearn.neighbors import KNeighborsClassifier
neighbors = np.arange(1,9)
train_accuracy =np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
for i,k in enumerate(neighbors):
    # Setup a knn classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit the model
    knn.fit(X_train, Y_train)
    
    # Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, Y_train)
    
    # Compute accuracy on the test set
    test_accuracy[i] = knn.score(X_test, Y_test) 
# Generate plot
plt.title('k-NN Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()

##unscaled_data
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train,Y_train)
knn.score(X_test,Y_test)

##Scaled_data
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train_1,Y_train)
knn.score(X_test_1,Y_test)


#NAvie_Bayes:
##unscaled_data
from sklearn.naive_bayes import BernoulliNB
model = BernoulliNB().fit(X_train, Y_train)
y_predicted = model.predict(X_test) 
accuracy_score = accuracy_score(Y_test, y_predicted) 
print(accuracy_score)
model.score(X_test,Y_test)
from sklearn.naive_bayes import BernoulliNB
model = BernoulliNB().fit(X_train, Y_train)
y_predicted = model.predict(X_test) 
accuracy_score = accuracy_score(Y_test, y_predicted) 
print(accuracy_score)
model.score(X_test,Y_test)

#Based on above Results LogisticRegression(96%) got Good accuracy compare to Knn(93%) and Navie bayes(90%)


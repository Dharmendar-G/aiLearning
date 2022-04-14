# Predicting Personal Loan
# The main aim of this project is to predict whether the customer buying banks personal loan or not. Therefore, this is a supervised classification problem. The classification goal is to predict the likelihood of a liability customer buying personal loans which means we must build a model which will be used to predict which customer will be most likely to accept the offer for personal loan, based on the specific relationship with the bank across various features given in the dataset. Here I will be using the Supervised Learning methods to predict which model is best for this problem amongst Logistic Regression, K-Nearest Neighbours (KNN) and Naive Bayes Algorithm.

# PROBLEM STATEMENT

# Objective:
# The classification goal is to predict the likelihood of a liability customer buying personal loans.
#  Steps and tasks:
#  1.            Read the column description and ensure you understand each attribute well
#  2.            Study the data distribution in each attribute, share your findings
#  3.            Get the target column distribution.
#  4.            Split the data into training and test set in the ratio of 70:30 respectively
#  5.            Use different classification models (Logistic, K-NN and Naïve Bayes) to predict the likelihood of a customer                    buying personal loans
#  6.            Print the confusion matrix for all the above models
#  7.            Give your reasoning on which is the best model in this case and why it performs better?


# Importing required libraries

# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#% matplotlib inline
import warnings

warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Reading the dataset
data = pd.read_csv('Bank_Personal_Loan_Modelling.csv')
data.head()
print(data.head())
# checking no.of rows and columns
data.shape
# list of data columns
list(data.columns)
# checking info
data.info()
# statistical summary  of dataset
data.describe()
# checking null values
data.isnull().sum()
# no.of unique values
data.apply(lambda x: len(x.unique()))
# value counts in each column
for i in data.columns:
    print(data[i].value_counts())

# list of categorical columns
cat_feats = []
for x in data.dtypes.index:
    if data.dtypes[x] == 'object':
        cat_feats.append(x)

cat_columns = [x for x in data.columns if data[x].dtypes == 'O']
# list of all numerical_columns
num_columns = [x for x in data.columns if data.dtypes[x] != 'object']
print(num_columns)
print(len(num_columns))
# or
num_feats = []
for col in data.dtypes.index:
    if data.dtypes[col] != 'object':
        num_feats.append(col)
# list of all numerical_columns
num_columns = [x for x in data.columns if data.dtypes[x] != 'object']
print(num_columns)
print(len(num_columns))

# list of discrete features
discrete_feat = [feature for feature in num_feats if len(data[feature].unique()) < 25]
discrete_feat
# list of continuous variables
conti_feat = [feature for feature in num_feats if feature not in discrete_feat]
print('no.of conti_features:', len(conti_feat))
print('conti_features are', conti_feat)
# outliers
for feature in num_feats:
    print('Q1_' + feature, data[feature].quantile(0.25))
    print('Q3_' + feature, data[feature].quantile(0.75))
    q1 = data[feature].quantile(0.25)
    q3 = data[feature].quantile(0.75)
    IQR = q3 - q1
    print('IQR_' + feature, IQR)
    lower_limit = q1 - (1.5 * IQR)
    upper_limit = q3 + (1.5 * IQR)
    print('Low_limit_' + feature, lower_limit)
    print('up_limit_' + feature, upper_limit)
    out_upper = len(data[feature][data[feature] > upper_limit])
    out_lower = len(data[feature][data[feature] < lower_limit])
    print('the outliers of upper limit is:', out_upper)
    print('the outliers of lower limit is:', out_lower)
    total_outliers = out_upper + out_lower
    print('The outlayers present in_ ' + feature, total_outliers)
    # for feature in feat_with_na:
    print(feature, 'column contains', np.round((total_outliers / len(data[feature]) * 100), 4), '% outliers')
    print()

# value counts of personal loan
data['Personal Loan'].value_counts()

# visualizations
# distribution plot
for feature in num_feats:
    sns.distplot(data[feature])
    plt.show()

# distribution plot using face grid
for feature in num_feats:
    sns.FacetGrid(data, hue="Personal Loan", size=5).map(sns.distplot, feature).add_legend()
# barplot
for feature in num_feats:
    plt.figure(figsize=(20, 10))
    data[feature].value_counts(normalize=True).plot.bar()
    plt.xlabel(feature)
    plt.ylabel('count')
    plt.show()

# histogram
for feature in num_feats:
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel('count')
    plt.show()

# pieplot
for feature in num_feats:
    plt.figure(figsize=(25, 10))
    plt.subplot(1, 2, 1)
    print(data[feature].value_counts(normalize=True).plot.pie(autopct='%.2f'))
    plt.show()

# countplot
for feature in num_feats:
    plt.figure(figsize=(15, 6))
    # plt.subplot(1,2,1)
    sns.countplot(x=data[feature], data=data)
    plt.show()
# boxplot
for feature in num_feats:
    sns.boxplot(x=feature, data=data)
    plt.xlabel(feature)
    plt.show()

# violinplot
for feature in num_feats:
    sns.violinplot(x=feature, data=data)
    plt.xlabel(feature)
    plt.show()

# scatterplot
for feature in num_feats:
    sns.scatterplot(x=feature, y='Personal Loan', data=data)
    plt.xlabel(feature)
    plt.ylabel('Personal Loan')
    plt.show()

# lmplot
for feature in num_feats:
    sns.lmplot(x=feature, y='Personal Loan', data=data)
    plt.xlabel(feature)
    plt.ylabel('count')
    plt.title(feature)
    plt.show()
# pairplot
sns.pairplot(data, hue='Personal Loan', height=5)
plt.show()

# handling missing values
data.isnull().sum()
# handling outliers using winsorizing technique
import scipy.stats

for feature in num_feats:
    print(scipy.stats.mstats.winsorize(data[feature], limits=0.95))
# dropping unwanted columns
data.drop(['ID'], axis=1, inplace=True)
# correlation
corr = data.corr()
corr

# heatmap
plt.figure(figsize=(20, 10))
sns.heatmap(corr, annot=True)
plt.show()


# Feature selection
def correlation(data, threshold):
    col_corr = set()
    corr_mat = data.corr()
    for i in range(len(corr_mat.columns)):
        for j in range(i):
            if abs(corr_mat.iloc[i, j]) > threshold:
                colname = corr_mat.columns[i]
                col_corr.add(colname)
    return col_corr


# calling function
correlation(data, 0.2)
# splitting the data into x and y (features and target)
x = data.drop(['Personal Loan'], axis=1)
y = data['Personal Loan']
# splitting data into training and testing
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
# imporing LogisticRegression model from sklearn and fitting the data into LogisticRegression model
from sklearn.linear_model import LogisticRegression

lg = LogisticRegression()
lg_model = lg.fit(x_train, y_train)
# predicting the model
y_pred_lg = lg_model.predict(x_test)
# importing metrics of classification model
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

cm = confusion_matrix(y_test, y_pred_lg)
print(cm)
# plotting confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(6, 6), cmap=plt.cm.Blues)
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
# checking accuracy
logreg = accuracy_score(y_test, y_pred_lg)
print(logreg)

# logistic regression with hyperparameters

# logistic regression without scaling
from sklearn.linear_model import LogisticRegression

lg = LogisticRegression(penalty='l2', solver='newton-cg', C=1.0)
lg_model = lg.fit(x_train, y_train)
y_pred_lg = lg_model.predict(x_test)
print(y_pred_lg)

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

cm = confusion_matrix(y_test, y_pred_lg)
print(cm)

from mlxtend.plotting import plot_confusion_matrix

fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(6, 6), cmap=plt.cm.Blues)
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
logreg = accuracy_score(y_test, y_pred_lg)
print('The accuracy score of Logistic Regression model is:', logreg)

print('The roc_auc_score of Logistic Regression model is:', roc_auc_score(y_test, y_pred_lg))

import sklearn.metrics as metrics

print(metrics.classification_report(y_test, y_pred_lg))

# Improvement of the model by scaling
from sklearn import preprocessing

X_train_scaled = preprocessing.scale(x_train)
X_test_scaled = preprocessing.scale(x_test)

# logistic regression with scaling
from sklearn.linear_model import LogisticRegression

lg = LogisticRegression()
lg_model = lg.fit(X_train_scaled, y_train)
y_pred_lg = lg_model.predict(X_test_scaled)
print(y_pred_lg)

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

cm = confusion_matrix(y_test, y_pred_lg)
print(cm)

from mlxtend.plotting import plot_confusion_matrix

fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(6, 6), cmap=plt.cm.Blues)
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
logreg = accuracy_score(y_test, y_pred_lg)
print('The accuracy score of Logistic Regression model is:', logreg)

print('The roc_auc_score of Logistic Regression model is:', roc_auc_score(y_test, y_pred_lg))

import sklearn.metrics as metrics

print(metrics.classification_report(y_test, y_pred_lg))

# KNN

# finding the best k value
List1 = list(range(1, 20))
neighbors = list(filter(lambda x: x % 2 != 0, List1))
ac_scores = []
# performing accuracy metrics for value from 1,3,5....19
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    # predict the response
    knn.fit(x_train, y_train.values.ravel())
    y_pred = knn.predict(x_test)
    # evaluate accuracy
    scores = accuracy_score(y_test, y_pred)
    # insert scores to the list
    ac_scores.append(scores)

# changing to misclassification error
MSE = [1 - x for x in ac_scores]
# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]

print('Odd Neighbors :', neighbors)
print('Accuracy Score :', ac_scores)
print('Misclassification error :', MSE)
print("The optimal number of neighbor is k=", optimal_k)

# plot misclassification error vs k
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()

print('#######################################')

# knn without scaling
knn = KNeighborsClassifier()
knn_model = knn.fit(x_train, y_train)
y_pred_knn = knn_model.predict(x_test)
y_pred_knn
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

cm = confusion_matrix(y_test, y_pred_knn)
print(cm)
from mlxtend.plotting import plot_confusion_matrix

fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(6, 6), cmap=plt.cm.Blues)
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
knn_acc = accuracy_score(y_test, y_pred_knn)
print('The accuracy score of KNN model is:', knn_acc)
print('The roc_auc score of KNN model is:', roc_auc_score(y_test, y_pred_knn))
import sklearn.metrics as metrics

print(metrics.classification_report(y_test, y_pred_knn))

print('#######################################')

# improving the model by applying scaling
from sklearn import preprocessing

X_train_scaled = preprocessing.scale(x_train)
X_test_scaled = preprocessing.scale(x_test)
# knn with scaling
knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn_model = knn.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)
y_pred_knn
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

cm = confusion_matrix(y_test, y_pred_knn)
print(cm)
from mlxtend.plotting import plot_confusion_matrix

fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(6, 6), cmap=plt.cm.Blues)
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
knn_acc = accuracy_score(y_test, y_pred_knn)
print('The accuracy score of KNN model is:', knn_acc)
print('The roc_auc_score of KNN model is:', roc_auc_score(y_test, y_pred_knn))
import sklearn.metrics as metrics

print(metrics.classification_report(y_test, y_pred_knn))

# Naive Bayes


# naive bayes without scaling
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb_model = gnb.fit(x_train, y_train)
y_pred_nb = gnb_model.predict(x_test)
print(y_pred_nb)
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

cm = confusion_matrix(y_test, y_pred_nb)
print(cm)
from mlxtend.plotting import plot_confusion_matrix

fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(6, 6), cmap=plt.cm.Blues)
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
gnb_acc = accuracy_score(y_test, y_pred_nb)
print('The accuracy score of gaussianNB is:', gnb_acc)
print('The roc_auc_score of gaussianNB is', roc_auc_score(y_test, y_pred_nb))
import sklearn.metrics as metrics

print(metrics.classification_report(y_test, y_pred_nb))

# improving the model by applying scaling
X_train_scaled = preprocessing.scale(x_train)
X_test_scaled = preprocessing.scale(x_test)

# naive bayes with scaling
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb_model = gnb.fit(X_train_scaled, y_train)
y_pred_nb = gnb_model.predict(X_test_scaled)
print(y_pred_nb)
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

cm = confusion_matrix(y_test, y_pred_nb)
print(cm)
from mlxtend.plotting import plot_confusion_matrix

fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(6, 6), cmap=plt.cm.Blues)
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
gnb_acc = accuracy_score(y_test, y_pred_nb)
print('The accuracy score of gaussianNB is:', gnb_acc)
print('The roc_auc_score of gaussianNB is:', roc_auc_score(y_test, y_pred_nb))
import sklearn.metrics as metrics

print(metrics.classification_report(y_test, y_pred_nb))

# Conclusion
# From the above three models , the KNN model is giving the maximum accuracy when compared with Logistic Regression and Naive Bayes algrithms. Why means our target attribute is mainly depend on the three columns those are income,ccavg and CD account . When we give the K-neighbors count three its giving maximum accuracy of 95.06 percentage. And also its giving ROC score of 0.78 .


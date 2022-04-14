#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt


# In[3]:



df = pd.read_csv('Bank_Personal_Loan_Modelling.csv')
df.columns


# In[4]:



df.head(10)


# In[5]:


column = ['ID', 'Age', 'Experience', 'Income', 'ZIP Code', 'Family', 'CCAvg',
       'Education', 'Mortgage', 'Personal Loan', 'Securities Account',
       'CD Account', 'Online', 'CreditCard']
data = pd.read_csv('Bank_Personal_Loan_Modelling.csv',names=column)


# In[6]:


data.shape


# In[7]:


df_loan = df


# In[8]:


y= df_loan['Personal Loan']
df_loan.drop(['Personal Loan'], axis = 1,inplace = True)
df['Personal Loan'] = y
df_loan.head(10)


# In[9]:


df_loan.info()


# In[10]:


df_loan.isnull().any()


# In[11]:


df_loan['Experience']


# Imputation is a process of replacing missing values with substituted values. In our dataset, some columns have missing values. We can replace missing values with mean, median, mode or any particular value. Sklearn provides Imputer() method to perform imputation in 1 line of code. We just need to define missing_values, axis, and strategy. We are using “median” value of the column to substitute with the missing value.

# In[12]:


df_loan[df_loan['Experience'] == -2]['Experience'].count()


# In[13]:


df_loan['Experience'].replace( to_replace= -1,value = np.nan,inplace = True )
df_loan['Experience'].replace( to_replace= -2,value = np.nan,inplace = True )
df_loan['Experience'].replace( to_replace= -3,value = np.nan,inplace = True )
df_loan['Experience'].isnull().sum()


# In[14]:


df_loan['Experience'].fillna(df_loan['Experience'].median(),inplace=True)
df_loan.describe().transpose()


# In[15]:


df.Experience.describe()


# In[16]:


df_loan.corr()


# In[17]:


df_loan["Personal Loan"].value_counts()# the aim here is a customer buying personal loans. hence target is personal loans


# In[18]:


sns.countplot(df_loan["Personal Loan"])


# In[19]:


#Findings 

#All the columns/attributes have 5000 non-null values.
#There is no null value present in the data frame.

#Total 52 negative values in Experience as Experience can't have negative values hence replacing it with a median

# ID:  This attribute can be dropped. Though the data distribution is normal.

# Age: Three small peaks can be indicating three values of age would be slightly more in number.

# Education: Mean and median are almost equal. 

# Income:  Data for the less income customers is more in the sample.

#  ZIP Code: The attribute has sharp peaks telling the data from particular places are collected more. .

# Family: It has 4 peaks(4 values), families with the least member is the highest in the sample.

# Mortgage: most customers are having least mortgage while very few have some mortgage.

# Securities Account:  majority of the customers are not having a Security account.

# CD account: Most of the customers don’t have CD accounts.

# Online: Higher number of customers use online banking in the sample.

#Credit Card: This attribute has fewer customers using CC in comparison to the CC users.



# In[21]:


count_no_buyers = len(df_loan[df_loan['Personal Loan']==0])
print('count_no_buyers :',count_no_buyers)
count_buyers = len(df_loan[df_loan['Personal Loan']==1])
print('count_buyers :',count_buyers)
pct_of_no_buyers = count_no_buyers/(count_no_buyers+count_buyers)
print('pct_of_no_buyers')
print("percentage of no buyers is", pct_of_no_buyers*100)
pct_of_buyers = count_buyers/(count_no_buyers+count_buyers)
print("percentage of buyers", pct_of_buyers*100)


# In[22]:


a = df_loan.var()
a[a<1]


# In[23]:


#Personal loans is having a comparatively better relation with Income.


# In[24]:


plt.scatter(df['Personal Loan'], df['Income'])
  
# Adding Title to the Plot
plt.title("Scatter Plot")
  
# Setting the X and Y labels
plt.xlabel('Personal Loan')
plt.ylabel('Income')
  
plt.show()


# In[25]:


plt.plot(df['Personal Loan'])
plt.plot(df['Income'])
  
# Adding Title to the Plot
plt.title("Scatter Plot")
  
# Setting the X and Y labels
plt.xlabel('Personal Loan')
plt.ylabel('Income')
  
plt.show()


# In[26]:


plt.bar(df['Personal Loan'], df['Income'])
  
plt.title("Bar Chart")
  
# Setting the X and Y labels
plt.xlabel('Personal Loan')
plt.ylabel('Income')
  
# Adding the legends
plt.show()


# In[27]:


plt.hist(df['Personal Loan'])
  
plt.title("Histogram")
  
# Adding the legends
plt.show()


# In[28]:


sns.lineplot(x="Personal Loan", y="Income", data=df)
  
# setting the title using Matplotlib
plt.title(' Personal Loan')
  
plt.show()


# In[29]:



sns.scatterplot(x='Personal Loan', y='Income', data=df,)
plt.show()


# In[30]:


sns.lineplot(data=df.drop(['Personal Loan'], axis=1))
plt.show()


# In[31]:


sns.lineplot(data=df.drop(['Experience'], axis=1))
plt.show()


# In[32]:


sns.lineplot(data=df.drop(['Income'], axis=1))
plt.show()


# In[33]:


sns.barplot(x='Personal Loan',y='Income', data=df, 
            hue='Experience')
  
plt.show()


# In[34]:


sns.histplot(x='Income', data=df, kde=True, hue='Personal Loan')
  
plt.show()


# In[35]:


df.plot.box()
  
# individual attribute box plot
plt.boxplot(df['Personal Loan'])
plt.show()


# In[36]:


df.plot.box()
  
# individual attribute box plot
plt.boxplot(df['Income'])
plt.show()


# In[37]:


from matplotlib import pyplot
fig, ax = pyplot.subplots(figsize =(7, 5))
sns.violinplot( ax = ax, y = df["Personal Loan"] )


# In[38]:


from matplotlib import pyplot
fig, ax = pyplot.subplots(figsize =(7, 5))
sns.violinplot( ax = ax, y = df["Experience"] )


# In[39]:


from matplotlib import pyplot
fig, ax = pyplot.subplots(figsize =(7, 5))
sns.violinplot( ax = ax, y = df["Income"] )


# In[40]:


df_loan.drop(columns ='Experience',inplace= True)
df_loan.drop(columns ='ID',inplace=True)
df_loan.head(2)


# In[41]:


sns.heatmap(df.isna(), yticklabels=False, cbar=False, cmap='viridis')


# In[42]:


df.nunique()


# In[43]:


sns.pairplot(df.iloc[:,1:])


# In[44]:


sns.distplot(df['Age'])


# In[45]:


sns.distplot(df['Income'])


# In[46]:


sns.distplot(df['ZIP Code'])


# In[47]:


sns.distplot(df['CCAvg'])


# In[48]:


sns.distplot(df['Education'])


# In[49]:


sns.distplot(df['Mortgage'])


# In[50]:


sns.distplot(df['Online'])


# In[51]:


sns.distplot(df['CreditCard'])


# In[52]:


loan_counts = pd.DataFrame(df["Personal Loan"].value_counts()).reset_index()
loan_counts.columns =["Labels","Personal Loan"]
loan_counts


# In[53]:


fig1, ax1 = plt.subplots()
explode = (0, 0.05)
ax1.pie(loan_counts["Personal Loan"], explode=explode, labels=loan_counts["Labels"], autopct='%1.1f%%',
        shadow=True, startangle=25)
ax1.axis('equal')  
plt.title("Personal Loan Percentage")
plt.show()


# In[54]:


sns.catplot(x='Family', y='Income', hue='Personal Loan', data = df, kind='swarm')


# In[55]:


sns.boxplot(x='Education', y='Income', hue='Personal Loan', data = df)


# In[56]:


sns.boxplot(x="Education", y='Mortgage', hue="Personal Loan", data=df)


# In[57]:


sns.countplot(x="Securities Account", data=df,hue="Personal Loan")


# In[58]:


sns.countplot(x='Family',data=df,hue='Personal Loan')


# In[59]:


sns.countplot(x='CD Account',data=df,hue='Personal Loan')


# In[60]:


sns.boxplot(x="CreditCard", y='CCAvg', hue="Personal Loan", data=df)


# In[61]:


sns.catplot(x='Age', y='Income', hue='Personal Loan', data = df, height=8.27, aspect=11/5)


# In[62]:


plt.figure(figsize=(10,4))
sns.distplot(df[df["Personal Loan"] == 0]['CCAvg'], color = 'r',label='Personal Loan=0')
sns.distplot(df[df["Personal Loan"] == 1]['CCAvg'], color = 'b',label='Personal Loan=1')
plt.legend()
plt.title("CCAvg Distribution")


# In[63]:


print('Credit card spending of Non-Loan customers: ',df[df['Personal Loan'] == 0]['CCAvg'].median()*1000)
print('Credit card spending of Loan customers    : ', df[df['Personal Loan'] == 1]['CCAvg'].median()*1000)


# In[64]:


plt.figure(figsize=(10,4))
sns.distplot(df[df["Personal Loan"] == 0]['Income'], color = 'r',label='Personal Loan=0')
sns.distplot(df[df["Personal Loan"] == 1]['Income'], color = 'b',label='Personal Loan=1')
plt.legend()
plt.title("Income Distribution")


# In[65]:


df.boxplot(return_type='axes', figsize=(15,5))


# In[66]:


plt.figure(figsize = (15,7))
plt.title('Correlation of Attributes', y=1.05, size=19)
sns.heatmap(df.corr(), cmap='plasma',annot=True, fmt='.2f')


# In[67]:


df.head(1)


# In[68]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(data.drop(['ID','Experience'], axis=1), test_size=0.3 , random_state=100)


# In[69]:


train_labels = train_set.pop('Personal Loan')
test_labels = test_set.pop('Personal Loan')


# In[70]:


array = df_loan
X= array.iloc[:,0:11]
y= array.iloc[:,11]


# In[71]:


standardized_X = preprocessing.scale(X)
standardized_X


# In[72]:


normalized_X = preprocessing.normalize(X)


# In[73]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)


# In[74]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[75]:


y_pred = logreg.predict(X_test)


# In[76]:


print('Accuracy of logistic regression classifier on train set: {:.2f}'.format(logreg.score(X_train, y_train)))


# In[77]:


print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# In[78]:


confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# In[79]:


print(classification_report(y_test, y_pred))


# In[80]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# In[81]:


import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())


# In[82]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)


# In[83]:


y_pred = classifier.predict(X_test)


# In[84]:


print('Accuracy of logistic regression classifier on train set: {:.2f}'.format(classifier.score(X_train, y_train)))


# In[85]:


print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))


# In[86]:


#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB


# In[87]:


#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = gnb.predict(X_test)


# In[88]:


print("Accuracy on test set:",metrics.accuracy_score(y_test, y_pred))


# In[89]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, square=True, annot=True, cmap='RdBu', cbar=False,xticklabels=['Yes', 'No'], yticklabels=['Yes' 'no'])
plt.xlabel('true label')
plt.ylabel('predicted label')


# In[90]:


print(classification_report(y_test, y_pred))


# In[91]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)


# In[92]:


y_pred = classifier.predict(X_test)


# In[93]:


print('Accuracy of logistic regression classifier on train set: {:.2f}'.format(classifier.score(X_train, y_train)))


# In[94]:


print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))


# In[95]:


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[96]:


error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))


# In[97]:


plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')


# # CONCLUSION

# The logistic Regression model is the best as the accuracy of the train and test set is almost similar and also the precsion and recall accuracy is good. The confusion matrix is also better in comparision to other models.
# 
# The requirement is to classify the target. The KNN is distance based which not perfect for this situation.Though the accuracy is good but confusion matrix tells that  correct predictions is not  much acceptable.
# 
# The Naive Bayes gives less accuracy when comapared to other models implying the probability of determing the correct target is less.

# In[ ]:





# In[ ]:





# # NOTES

# A.Logistic Regression is a Machine Learning classification algorithm that is used to predict the probability of a categorical dependent variable. In logistic regression, the dependent variable is a binary variable that contains data coded as 1 (yes, success, etc.) or 0 (no, failure, etc.). In other words, the logistic regression model predicts P(Y=1) as a function of X.
# Logistic Regression Assumptions
# Binary logistic regression requires the dependent variable to be binary.
# For a binary regression, the factor level 1 of the dependent variable should represent the desired outcome.
# Only the meaningful variables should be included.
# The independent variables should be independent of each other. That is, the model should have little or no multicollinearity.
# The independent variables are linearly related to the log odds.
# Logistic regression requires quite large sample sizes
# 
# 
# Logistic regression becomes a classification technique only when a decision threshold is brought into the picture. The setting of the threshold value is a very important aspect of Logistic regression and is dependent on the classification problem itself.
# 
# The decision for the value of the threshold value is majorly affected by the values of precision and recall
# we use the following arguments to decide upon the threshold:-
# 1.Low Precision/High Recall
# 2.High Precision/Low Recall
# 
# 

# B. A Confusion matrix is an N x N matrix used for evaluating the performance of a classification model, where N is the number of target classes. The matrix compares the actual target values with those predicted by the machine learning model.
# The target variable has two values: Positive or Negative
# The columns represent the actual values of the target variable
# The rows represent the predicted values of the target variable
# True Positive (TP) 
# True Negative (TN) 
# False Positive (FP) – Type 1 error
# False Negative (FN) – Type 2 error
#  Precision is the ratio between the True Positives and all the Positives. 
# Precision is a useful metric in cases where False Positive is a higher concern than False Negatives
# 
# Recall quantifies the number of positive class predictions made out of all positive examples in the dataset
# 
# Recall is a useful metric in cases where False Negative trumps False Positive.
# 
# The confusion matrix provides more insight into not only the performance of a predictive model, but also which classes are being predicted correctly, which incorrectly, and what type of errors are being made

# C. A classification problem has a discrete value as its output. For example, “likes pineapple on pizza” and “does not like pineapple on pizza” are discrete
# A supervised machine learning algorithm (as opposed to an unsupervised machine learning algorithm) is one that relies on labeled input data to learn a function that produces an appropriate output when given new unlabeled data.
# Supervised machine learning algorithms are used to solve classification or regression problems.
# EX: When we see a pig, we shout “pig!” When it’s not a pig, we shout “no, not pig!” After doing this several times with the child, we show them a picture and ask “pig?” and they will correctly (most of the time) say “pig!” or “no, not pig!” depending on what the picture is. That is supervised machine learning.
# 
# 
# 
# The k-nearest neighbors (KNN) algorithm is a simple, easy-to-implement supervised machine learning algorithm that can be used to solve both classification and regression problems.
# The KNN algorithm assumes that similar things exist in close proximity. In other words, similar things are near to each other.

# D. Naive Bayes Algorithm  is a classification technique based on Bayes’ Theorem with an assumption of independence among predictors. In simple terms, a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature.
# 
# For example, a fruit may be considered to be an apple if it is red, round, and about 3 inches in diameter. Even if these features depend on each other or upon the existence of the other features, all of these properties independently contribute to the probability that this fruit is an apple and that is why it is known as ‘Naive’.
# 
# Naive Bayes model is easy to build and particularly useful for very large data sets. Along with simplicity, Naive Bayes is known to outperform even highly sophisticated classification methods.

# In[ ]:





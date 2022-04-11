#!/usr/bin/env python
# coding: utf-8

# In[66]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
sns.set(style="whitegrid")
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.preprocessing import RobustScaler

df=pd.read_csv('Dataset_Usecase-5.csv')

df.head()

df.shape

df.info()

df.columns

### The names of the columns are not proper (having spaces, '-' etc)
#### Setting up the names

edited_cols = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
             'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
df.columns = edited_cols

df.columns

## Categorical Columns

cat_columns=[col for col in df.columns if df[col].dtype=='object']

cat_columns

for val in cat_columns:
    print('Columns Name: ----------------------------------%s'%val)
    print(df[val].value_counts())

#### in this case the missing values are coded as "?"

##### 

## Numerical Columns

num_columns=[col for col in df.columns if df[col].dtype=='int64']

num_columns

df.describe().T

## Target Column

df['income'].value_counts()

df['income'].value_counts()/len(df)



## Data Engineering and EDA

### Categorical Columns

#### work_class

df['workclass'].value_counts()

df.columns

df['workclass'].replace(' ?',np.NaN,inplace=True)

df['workclass'].value_counts()

plt.figure(figsize=(14,6))
ax=df['workclass'].value_counts().plot(kind='bar',color='brown')
ax.set_title('Frequency Distribution of WorkClass')

##### More Private workers compared to anyother

plt.figure(figsize=(14,6))
ax=sns.countplot(x='workclass',hue='income',data=df)
ax.set_title('Workclass wrt to Income')
ax.legend()

#### Most of the people are earning <=50k

#### occupation

df['occupation'].value_counts()

df['occupation'].replace(' ?',np.NaN,inplace=True)

df['occupation'].value_counts()

plt.figure(figsize=(14,6))
ax=df['occupation'].value_counts().plot(kind='bar',color='orange')
ax.set_title('Frequency Distribution of Occupation')

plt.figure(figsize=(16,6))
ax=sns.countplot(x='occupation',hue='income',data=df,palette='inferno')
ax.set_title('Occupation wrt to Income')
ax.set_xticklabels(df.occupation.value_counts().index, rotation=30)
ax.legend()

plt.figure(figsize=(16,6))
ax=sns.countplot(x='occupation',hue='sex',data=df,palette='turbo')
ax.set_title('Occupation wrt to Sex')
ax.set_xticklabels(df.occupation.value_counts().index, rotation=30)
ax.legend()

#### Most of the occuaptions are dominated by Males
##### Sales and Prof-speciality are the regions where females are more

#### martial_status

df['marital_status'].value_counts()

#### relationship

df['relationship'].value_counts()

#### race

df['race'].value_counts()

plt.figure(figsize=(12,6))
ax=sns.countplot(x='race',hue='income',data=df,palette='cool_r')
ax.set_title('Race wrt to Income')
ax.legend()

plt.figure(figsize=(16,6))
ax=sns.countplot(x='race',hue='workclass',data=df,palette='prism')
ax.set_title('Race wrt to Workclass')
ax.legend(loc='upper right')

#### native country

df['native_country'].value_counts()

df['native_country'].replace(' ?',np.NaN,inplace=True)

df['native_country'].value_counts()[:8]

#### the data we have is more domianted with 'UnitedStates'

## Numerical Columns

num_columns

#### age

df['age'].unique()

plt.figure(figsize=(12,6))
ax = sns.distplot(df['age'], bins=10, color='blue')
ax.set_title("Distribution of age variable")
plt.show()

plt.figure(figsize=(14,5))
ax = sns.boxplot(df['age'],palette='hsv')
ax.set_title("Visualize outliers in age variable")
plt.show()

###### we can find outliers in the age data

f, ax = plt.subplots(figsize=(10, 8))
ax = sns.boxplot(x="income", y="age", data=df,palette='OrRd')
ax.set_title("Visualize income wrt age variable")
plt.show()

###### as the age increases, salary also increases

plt.figure(figsize=(12,8))
sns.boxplot(x ='race', y="age", data = df,palette='autumn')
plt.title("Visualize age wrt race")
plt.show()

###### whites are elder than others

##### education_num

df['education_num'].unique()

plt.figure(figsize=(12,6))
ax=sns.countplot(x='education_num',hue='income',data=df,palette='flare')
ax.set_title('Education wrt to Income')
ax.legend()

###### people with less than 9 grade are unable to earn >50k

###### hours_per_weak

df['hours_per_week'].describe()

##### average working hours per week are 40hours

sns.pairplot(df, hue="income")
plt.show()

###### age and fnlwgt are positively skewed
###### education_num, hours_per_week are normaly distributed

X=df.drop(['income'],axis=1)
y=df['income']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

X_train.shape, X_test.shape

plt.hist(y)

#### there are some missing values in the dataset
##### Imputing them with mode for cat features

X_train.isnull().sum()

for df2 in [X_train, X_test]:
    df2['workclass'].fillna(X_train['workclass'].mode()[0], inplace=True)
    df2['occupation'].fillna(X_train['occupation'].mode()[0], inplace=True)
    df2['native_country'].fillna(X_train['native_country'].mode()[0], inplace=True) 

X_train.isnull().sum()

## categorical to numerical

encoder = ce.OneHotEncoder(cols=['workclass', 'education', 'marital_status', 
                                 'occupation', 'relationship','race', 
                                 'sex', 'native_country'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)

X_train.head()

X_train.shape, X_test.shape

## Feature Scaling

cols=X_train.columns

scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])

## Model Building

#### XGB

from xgboost import XGBClassifier

xgb=XGBClassifier()

y_pred = xgb.fit(X_train, y_train).predict(X_test)

from sklearn.metrics import accuracy_score

ac=accuracy_score(y_test,y_pred)
ac

### Random Forest

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100,random_state=0)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

rac=accuracy_score(y_test,y_pred)
rac


import numpy as np
import pandas as pd
import pickle
from flask import Flask, render_template, make_response, request
import io
from io import StringIO
from xgboost import XGBClassifier

app=Flask(__name__)

def feature_engineering(df):

    #df.columns=['age', 'workclass', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'hours_per_week', 'native_country', 'income']
    
    df.columns=df.columns.str.replace(' ','')
    df.columns=df.columns.str.replace('-','_')    
    
    #workclass
    workClass={value:key for key,value in enumerate(df['workclass'].unique())}
    df['workclass']=df['workclass'].map(workClass)
    
    #marital_status
    maritalStatus={value:key for key,value in enumerate(df['marital_status'].unique())}
    df['marital_status']=df['marital_status'].map(maritalStatus)
    
    #occupation
    occuPation={value:key for key,value in enumerate(df['occupation'].unique())}
    df['occupation']=df['occupation'].map(occuPation)
    
    #relationship
    relationShip={value:key for key,value in enumerate(df['relationship'].unique())}
    df['relationship']=df['relationship'].map(relationShip)
    
    #race
    Race={value:key for key,value in enumerate(df['race'].unique())}
    df['race']=df['race'].map(Race)
    
    #sex
    Sex={value:key for key,value in enumerate(df['sex'].unique())}
    df['sex']=df['sex'].map(Sex)
    
    #native_country
    nativeCountry={value:key for key,value in enumerate(df['native_country'].unique())}
    df['native_country']=df['native_country'].map(nativeCountry)
    
    return df

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    f=request.files['data_file']
    if not f:
        return render_template('index.html',prediction_text='No file selected')

    stream=io.StringIO(f.stream.read().decode("UTF8"),newline=None)
    result=stream.read()

    df=pd.read_csv(StringIO(result))

    df=feature_engineering(df)

    X=df[['age', 'workclass', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'hours_per_week', 'native_country']]

    loaded_model=pickle.load(open("xgbModel.pkl",'rb'))

    result=loaded_model.predict(X)

    return render_template('index.html',prediction_text='Predicted Salary is: {}'.format(result))

if __name__=="__main__":
    app.run(debug=False,port=9000)
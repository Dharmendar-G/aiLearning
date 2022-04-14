import streamlit as st
import pickle
import numpy as np
from scipy.sparse import hstack
import scipy.sparse as sp
from sklearn.ensemble import  GradientBoostingClassifier
from xgboost import XGBClassifier, Booster

grad = pickle.load(open("random.pkl", 'rb'))

def pred(age,workclass,education,marital_status,occupation,relation,race,gender,hours,country):
    # age = np.array(age).reshape(1, -1)
    # workclass= np.array(workclass).reshape(1, -1)
    # education= np.array(education).reshape(1, -1)
    # marital_status= np.array(marital_status).reshape(1, -1)
    # occupation=np.array(occupation).reshape(1, -1)
    # relation= np.array(relation).reshape(1, -1)
    # race= np.array(race).reshape(1, -1)
    # gender= np.array(gender).reshape(1, -1)
    # hours= np.array(hours).reshape(1, -1)
    # country= np.array(country).reshape(1, -1)

    # input = hstack((age,workclass,education,marital_status,occupation,relation,race,gender,hours,country)).tocsr()

    out = grad.predict([[age,workclass,education,marital_status,occupation,relation,race,gender,hours,country]])
    return out

def main():
    htm="""
    <div style="background-color:Gold;padding:9px">
    <h2 style="color:white;text-align:center;">Income Class Predictor</h2>
    </div>
    """
    st.markdown(htm,unsafe_allow_html=True)

    #st.write("Please fill the fields and make the magic happen!!!")


    st.subheader('Enter the details:')
    a=1

    with st.form("my_form"):
        age = st.number_input('Age(0-99) :')

        workclass = st.selectbox('Work Class:',(' State-gov',
                                                ' Self-emp-not-inc',
                                                ' Private',
                                                ' Federal-gov',
                                                ' Local-gov',
                                                ' Self-emp-inc',
                                                ' Without-pay',
                                                ' Never-worked'))

        education = st.selectbox('Education:',(' Bachelors', ' HS-grad', ' 11th', ' Masters', ' 9th',
       ' Some-college', ' Assoc-acdm', ' Assoc-voc', ' 7th-8th',
       ' Doctorate', ' Prof-school', ' 5th-6th', ' 10th', ' 1st-4th',
       ' Preschool', ' 12th'))

        marital_status = st.selectbox('Marital Status:',(' Never-married',
                                                         ' Married-civ-spouse',
                                                         ' Divorced',
                                                         ' Married-spouse-absent',
                                                         ' Separated',
                                                         ' Married-AF-spouse',
                                                         ' Widowed'))

        occupation = st.selectbox('Occupation:',(' Adm-clerical',
 ' Exec-managerial',
 ' Handlers-cleaners',
 ' Prof-specialty',
 ' Other-service',
 ' Sales',
 ' Craft-repair',
 ' Transport-moving',
 ' Farming-fishing',
 ' Machine-op-inspct',
 ' Tech-support',
 ' Protective-serv',
 ' Armed-Forces',
 ' Priv-house-serv'))


        relation = st.selectbox('Relation:', (' Not-in-family',
 ' Husband',
 ' Wife',
 ' Own-child',
 ' Unmarried',
 ' Other-relative'))

        race=st.selectbox("Race:",(' White', ' Black', ' Asian-Pac-Islander', ' Amer-Indian-Eskimo', ' Other'))

        gender=st.selectbox("Gender:",('Male', 'Female'))

        hours=st.number_input("Working hours per week:")

        country=st.selectbox("Country:",(' United-States', 'other_country'))

        submitted = st.form_submit_button("Predict")


        workclass_dic={' Federal-gov': 3,' Local-gov': 4,' Never-worked': 7, ' Private': 2,' Self-emp-inc': 5,' Self-emp-not-inc': 1, ' State-gov': 0, ' Without-pay': 6}
        workclass= workclass_dic[workclass]

        education_dic={' 10th': 6,' 11th': 7,' 12th': 8, ' 1st-4th': 2, ' 5th-6th': 3, ' 7th-8th': 4, ' 9th': 5, ' Assoc-acdm': 12, ' Assoc-voc': 11, ' Bachelors': 13, ' Doctorate': 16, ' HS-grad': 9, ' Masters': 14, ' Preschool': 1, ' Prof-school': 15, ' Some-college': 10}
        education=education_dic[education]

        marital_dic={' Divorced': 2,' Married-AF-spouse': 5,' Married-civ-spouse': 1, ' Married-spouse-absent': 3,' Never-married': 0, ' Separated': 4,' Widowed': 6}
        marital_status=marital_dic[marital_status]

        occupation_dic={' Adm-clerical': 0,
 ' Armed-Forces': 12,
 ' Craft-repair': 6,
 ' Exec-managerial': 1,
 ' Farming-fishing': 8,
 ' Handlers-cleaners': 2,
 ' Machine-op-inspct': 9,
 ' Other-service': 4,
 ' Priv-house-serv': 13,
 ' Prof-specialty': 3,
 ' Protective-serv': 11,
 ' Sales': 5,
 ' Tech-support': 10,
 ' Transport-moving': 7}
        occupation=occupation_dic[occupation]


        relation_dic={' Husband': 1,
 ' Not-in-family': 0,
 ' Other-relative': 5,
 ' Own-child': 3,
 ' Unmarried': 4,
 ' Wife': 2}
        relation=relation_dic[relation]


        race_dic={' Amer-Indian-Eskimo': 3,
 ' Asian-Pac-Islander': 2,
 ' Black': 1,
 ' Other': 4,
 ' White': 0}
        race=race_dic[race]

        gender_dic={"Male":0,"Female":1}
        gender=gender_dic[gender]

        age=int(age//1)
        hours=int(hours//1)

        country_dic={' United-States': 0, 'other_country': 1}
        country=country_dic[country]

        #inp=[age,workclass,education,marital_status,occupation,relation,race,gender,hours,country]
        out=pred(age,workclass,education,marital_status,occupation,relation,race,gender,hours,country)
        out.ravel()
        out=list(out)
        out=out[0]

        if submitted:
            a=0

    if a == 0:
        st.write("Prediction of income is  ",'"',out,'"')
if __name__=="__main__":
    main()
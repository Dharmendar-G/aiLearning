
from flask import Flask,render_template,request
import joblib
import numpy as np


app=Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')
 

@app.route("/predict",methods=['POST'])   
def result():
    if request.method=='POST':
    
        n1=request.form.get("N")
        n2=request.form.get("P")
        n3=request.form.get("K")
        n4=request.form.get("temp")
        n5=request.form.get("Humidity")
        n6=request.form.get("ph")
        n7=request.form.get("rainfall") 

    
        data=np.array([[float(n1),float(n2),float(n3),float(n4),float(n5),float(n6),float(n7)]]).reshape(1,7)
        import joblib
        rf_model = joblib.load('rf.pkl')  
        result=rf_model.predict(data)
        # print(result)
        # result=rf.predict(np.array([float(n1),float(n2),float(n3),float(n4),float(n5),float(n6),float(n7)]).reshape(1,7))
    return render_template('index.html',result=result)    

if __name__=='__main__':

      app.run(host='0.0.0.0',port=8080)


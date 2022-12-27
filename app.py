from flask import Flask,url_for,render_template,request
import pickle
import numpy as np

model=pickle.load(open('model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def welcome():
    return render_template('body.html')

@app.route('/submit',methods=['POST'])
def predict():
    if request.method=='POST':
        age=float(request.form['age'])
        gender=float(request.form['gender'])
        cp=float(request.form['cp'])
        trestbps=float(request.form['trestbps'])
        chol=float(request.form['chol'])
        fbs=float(request.form['fbs'])
        restecg=float(request.form['restecg'])
        thalach=float(request.form['thalach'])
        exang=float(request.form['exang'])
        oldpeak=float(request.form['oldpeak'])
        slope=float(request.form['slope'])
        ca=float(request.form['ca'])
        thal=float(request.form['thal'])
        a=np.array([[age,gender,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
    prediction=model.predict(a)
    if prediction==1:
        return render_template('result1.html')
    else:
        return render_template('result0.html')


if __name__=="__main__":
    app.run(debug=True)
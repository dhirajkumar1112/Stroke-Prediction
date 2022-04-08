from flask import * 
from flask import Flask,flash, session, render_template, request, redirect
from tensorflow import keras
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.preprocessing import LabelEncoder

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


app = Flask(__name__) 
app.secret_key = b'_DKRP/'

model = keras.models.load_model('model.h5')

@app.route('/')
def index():
      return render_template('index.html')

@app.route('/predict',methods = ['POST'])  
def predict():
      #global loginSts
      idd=int(request.form['id'])
      gender=request.form['gender']
      age=int(request.form['age'])
      hypertension=int(request.form['hypertension'])
      heart_disease=int(request.form['heart_disease'])
      ever_married=request.form['ever_married']
      work_type=request.form['work_type']
      Residence_type=request.form['Residence_type']
      avg_glucose_level=float(request.form['avg_glucose_level'])
      bmi=float(request.form['bmi'])
      smoking_status=request.form['smoking_status']


      data={"id":[123,456,789,110,111,idd],"gender":["Male","Female","Male","Female","Male",gender],"age":[67,30,45,27,15,age],"hypertension":[1,0,1,1,0,hypertension],"heart_disease":[1,0,1,1,0,heart_disease],"ever_married":["Yes","No","Yes","Yes","No",ever_married],"work_type":["Private","Self-employed","Govt_job","Never_worked","children",work_type],"Residence_type":["Urban","Rural","Urban","Urban","Urban",Residence_type],"avg_glucose_level":[230.67,202.21,186.21,161.28,114.84,avg_glucose_level],"bmi":[36.6,32.5,34.4,41.8,25.7,bmi],"smoking_status":["smokes","never smoked","formerly smoked","Unknown","formerly smoked",smoking_status]}

      df = pd.DataFrame(data)


      df.drop('id', axis = 1, inplace = True)
      df = df[df['age'] > 13]

      temp_columns = pd.get_dummies(df.work_type)
      df.drop('work_type', axis = 1, inplace = True)
      temp_df1 = df[[column for column in df.columns[:5]]]
      temp_df1 = pd.concat([temp_df1, temp_columns], axis = 1)
      temp_df2 = pd.get_dummies(df.smoking_status)
      temp_df3 = df[[column for column in df.columns[5:8]]]
      temp_df1 = pd.concat([temp_df1, temp_df3, temp_df2,], axis = 1)

      # Encoding the 'gender', 'ever_married' and 'Residence_type' attributes using label encoding
      label_encoder = LabelEncoder()

      temp_df1['gender'] = label_encoder.fit_transform(temp_df1['gender'])
      temp_df1['ever_married'] = label_encoder.fit_transform(temp_df1['ever_married'])
      temp_df1['Residence_type'] = label_encoder.fit_transform(temp_df1['Residence_type'])

      df = temp_df1

      data_scaler = MinMaxScaler()
      df_scaled = data_scaler.fit_transform(df)

      prediction = model.predict(df_scaled)
      prediction = np.round(prediction)
      print(prediction)

      pred_val = prediction[-1][0]
      print(pred_val)
      if (pred_val==1.0):
            status = 'You have chances of getting a Stroke'
      else:
            status = "Don't worry your fine! "


      return render_template('index.html',status = status)


if __name__ == '__main__':
    app.run(debug = True) 
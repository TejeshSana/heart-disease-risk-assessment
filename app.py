import numpy as np
from flask import Flask, request, render_template
import pickle
from pymongo import MongoClient

app = Flask(__name__)

# Load the scaler and model
sc = pickle.load(open('sc.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# MongoDB connection
client = MongoClient("mongodb://localhost:27017")
db = client["heart_disease_db"]
collection = db["predictions"]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    lst = []
    cp = int(request.form['chest pain type (4 values)'])
    if cp == 0:
        lst += [1, 0, 0, 0]
    elif cp == 1:
        lst += [0, 1, 0, 0]
    elif cp == 2:
        lst += [0, 0, 1, 0]
    elif cp >= 3:
        lst += [0, 0, 0, 1]
    
    trestbps = int(request.form["resting blood pressure"])
    lst += [trestbps]
    
    chol = int(request.form["serum cholestoral in mg/dl"])
    lst += [chol]
    
    fbs = int(request.form["fasting blood sugar > 120 mg/dl"])
    if fbs == 0:
        lst += [1, 0]
    else:
        lst += [0, 1]
    
    restecg = int(request.form["resting electrocardiographic results (values 0,1,2)"])
    if restecg == 0:
        lst += [1, 0, 0]
    elif restecg == 1:
        lst += [0, 1, 0]
    else:
        lst += [0, 0, 1]
    
    thalach = int(request.form["maximum heart rate achieved"])
    lst += [thalach]
    
    exang = int(request.form["exercise induced angina"])
    if exang == 0:
        lst += [1, 0]
    else:
        lst += [0, 1]
    
    final_features = np.array([lst])
    pred = model.predict(sc.transform(final_features))
    
    # Save the prediction and user input to MongoDB
    data = {
        "input": {
            "chest_pain_type": int(cp),
            "resting_blood_pressure": int(trestbps),
            "serum_cholestoral": int(chol),
            "fasting_blood_sugar": int(fbs),
            "resting_electrocardiographic": int(restecg),
            "maximum_heart_rate": int(thalach),
            "exercise_induced_angina": int(exang)
        },
        "prediction": int(pred[0])
    }
    collection.insert_one(data)
    
    return render_template('results.html', prediction=pred)

if __name__ == "__main__":
    app.run(debug=True)

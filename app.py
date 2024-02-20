from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)
with open('Models\social_Ads_classifier_RF.pickle', 'rb') as file:
    Social_ads_classifier = pickle.load(file)

with open('Models\social_Ads_gender_preprocessor.pickle', 'rb') as file:
    gender_lables_encoder = pickle.load(file)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    Gender = request.form.get('Gender')
    Gender = gender_lables_encoder.transform(np.array([Gender]))[0]
    Age = request.form.get('Age')
    Salary = request.form.get('Salary')
    features = [Gender, Age, Salary]
    int_features = [int(x) for x in features]
    predict_features = [np.array(int_features)]
    prediction = Social_ads_classifier.predict(predict_features) 
    if prediction[0] == 1:
        predicted_output = 'likely to make a purchase'
    elif prediction[0]==0:
        predicted_output = 'likely to make a purchase'

    print('Purchaser', predict_features, prediction, predicted_output)
    return render_template('Predict.html', prediction_result=predicted_output)

@app.route('/predict', methods=['GET'])
def pre():
    return render_template('Predict.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)


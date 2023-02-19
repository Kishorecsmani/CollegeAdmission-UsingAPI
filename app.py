import pickle
import flask 
import pandas as pd

from flask import Flask, request, jsonify
app = Flask(__name__)

# loading the ml model
model_pickle = open("./linear_education_model.pkl","rb")
scaler_pickle = open("./linear_education_scaler.pkl","rb")
clf = pickle.load(model_pickle)
scaler = pickle.load(scaler_pickle)

# 2nd
@app.route("/predictedu", methods=['POST'])
def prediction():

    edu_req = request.get_json()
    #print(edu_req)
    edu_app_sc = pd.DataFrame(edu_req, index=[1,])
    X_train_columns=edu_app_sc.columns
    #print(edu_app_sc)
    edu_app_sc1 = scaler.transform(edu_app_sc)
    edu_app_sc1 = pd.DataFrame(edu_app_sc1, columns=X_train_columns)
    edu_app_sc1  = edu_app_sc1 .round(2)
    #print(edu_app_sc1)
    edu_final = edu_app_sc1.T.to_dict()[0]
    #print(edu_final)
    gre_score = edu_final['GRE_Score']
    toefl_score =  edu_final['TOEFL_Score']
    univ_rating =  edu_final['University_Rating']
    sop =  edu_final['SOP']
    lor =  edu_final['LOR']
    cgpa =  edu_final['CGPA']
    research =  edu_final['Research']
    input_data = [[gre_score, toefl_score, univ_rating, sop, lor, cgpa, research]]
    # generate inference
    pred = clf.predict(input_data).round(2)
    #print(pred[0]) 
    return {"chances of admit" : pred[0]}
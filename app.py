import pickle
import flask 

from flask import Flask, request
app = Flask(__name__)

# loading the ml model
model_pickle = open("./linear_education_model.pkl","rb")
scaler_pickle = open("./linear_education_scaler.pkl","rb")
clf = pickle.load(model_pickle)
scaler = pickle.load(scaler_pickle)

# 2nd
@app.route("/predictedu", methods=["POST"])
def prediction():
    edu_req = request.get_json()
    
    gre_score = edu_req['GRE_Score']
    toefl_score =  edu_req['TOEFL_Score']
    univ_rating =  edu_req['University_Rating']
    sop =  edu_req['SOP']
    lor =  edu_req['LOR']
    cgpa =  edu_req['CGPA']
    research =  edu_req['Research']
    input_data = [[gre_score, toefl_score, univ_rating, sop, lor, cgpa, research]]
    # generate inference
    pred = clf.predict(input_data) 
    return {"chances of admit" : pred}
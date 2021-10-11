from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_json import FlaskJSON, JsonError, json_response, as_json

##import C:\Users\AKHALBOUS\Desktop\code_python\encoder_one_hot.npy as encoder_one_hot


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pickle
import re
from numpy import linalg as la




app = Flask(__name__)
FlaskJSON(app)


def encodage(vect, encoder1, encoder2, encoder3):
    aux = encoder2.transform(vect[one_hot_columns])
    
    vect = vect.drop(columns =one_hot_columns, axis=1)
    vect["origine"] = encoder1.transform(vect["origine"])
    vect[['mois',"age"]] = encoder3.transform(vect[[ 'mois',"age"]])
    vect[[ 'mois',"age"]] = encoder3.transform(vect[[ 'mois',"age"]])
    
    res= pd.concat([vect.reset_index(drop = True), pd.DataFrame(aux.toarray()).reset_index(drop = True)], axis=1)
    return res
	
def multi(n,model,x_test,encoder):
    y_pred = model.predict_proba(x_test)
    x = np.arange(y_pred.shape[0]*n).reshape(y_pred.shape[0],n)
    z = []
    p = np.arange(y_pred.shape[0]*n).reshape(y_pred.shape[0],n)
    for i in range(y_pred.shape[0]):
        x[i] = y_pred[i].argsort()[-n:][::-1]
        z.append(encoder.inverse_transform(x[i]))
        p[i] = (np.sort(y_pred[i])[-n:][::-1]) *100
    p = ''.join(map(str, p))
    z = ''.join(map(str, z))
    res = '\n les offres les plus probables sont : {} \n \n avec les probabilités suivantes : {} \n'.format(z,p)
    
    return res
	
def offre(res,prob,vect,n,encoder):
    if res == 1:
        filename = 'model_multiclass_XGB.sav'
        model = pickle.load(open(filename, 'rb'))
        a1 = ' \n la probabilité de acceptation est de {} %  \n \n le prospect aura tendance à accepter cette proposition \n'.format(prob[0][1]*100)
        a2 = multi(n,model,vect,encoder)
        a3 = a1 + a2
    else :
        a3 = "\n la probabilité de refus est de {} % \n \n le prospect aura tendance à refuser cette proposition \n ".format(prob[0][0]*100) 
    return a3
		
def prospection(vect,encoder):
    if (vect["age"] < 30).any():
        filename = 'model_inferieur_q1_xgb.sav'
        model = pickle.load(open(filename, 'rb'))
        pred = model.predict(vect)
        prob = model.predict_proba(vect)
        res =  int(pred)
        b =offre(res,prob,vect,3,encoder)
        
    else :
        filename = 'model_superieur_q1_xgb.sav'
        model = pickle.load(open(filename, 'rb'))
        pred = model.predict(vect)
        prob = model._predict_proba_lr(vect)
        res = int(pred)
        b = offre(res,prob,vect,3,encoder)
    return b
		

one_hot_columns = ['SEXE','CIVILITÉ','SITUATION_FAMILIALE','CANAL_ORIGINE','adherent','departement',"GAMME_PROPOSITION"]
encoder_hot = np.load('encoder_one_hot.npy', allow_pickle=True).tolist()
encoder_label = np.load('encoder_label.npy', allow_pickle=True).tolist()
encoder_multi = np.load('encoder_multi.npy', allow_pickle=True).tolist()
normalizer = np.load('normalizer.npy', allow_pickle=True).tolist()		
df = pd.read_csv(r"C:\Users\AKHALBOUS\Desktop\code_python\df_api_2.csv",sep=',',index_col=0)

#prospection(encodage(df.iloc[32:33],encoder_label, encoder_hot, normalizer),encoder_multi)

@app.route('/')
def home():
    return render_template('index.html')


	
@app.route('/predict',methods=['POST'])
def predict():
   
    d = request.form.to_dict()
    vect = pd.DataFrame([d.values()], columns=d.keys())
    vect["departement"] = int( vect["departement"])
    vect["age"] = int( vect["age"])
    vect["mois"] = int( vect["mois"])
    vect["STATUT_ASSURÉ_RC"] = int( vect["STATUT_ASSURÉ_RC"])
    vect["STATUT_ASSURÉ_RO"] = int( vect["STATUT_ASSURÉ_RO"])
    output = prospection(encodage(vect,encoder_label, encoder_hot, normalizer),encoder_multi)

    return render_template('index.html', prediction_text= output)

@app.route('/Marketing', methods=['POST'])
def Marketing():
	data = request.get_json(force=True)
	try:
	
		vect = pd.DataFrame.from_dict(pd.json_normalize(data), orient="columns")
		res = prospection(encodage(vect,encoder_label, encoder_hot, normalizer),encoder_multi)
		print(res)
		return('')

    
	except (KeyError, TypeError, ValueError):
		raise JsonError(description='Invalid value.')
	
	
	
if __name__ == '__main__':
	app.run(debug=True)	
		
		
		
### https://github.com/krishnaik06/Deployment-flask

	
	

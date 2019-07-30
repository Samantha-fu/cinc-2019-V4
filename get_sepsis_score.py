import numpy as np
import pandas as pd
from sklearn.externals import joblib
AB_features_mean_dict={'AST': 356.2075296108291,
 'Age': 63.0167798510459,
 'Alkalinephos': 114.20330385015609,
 'BUN': 24.346708852906506,
 'BaseExcess': -0.6475370534467899,
 'Bilirubin_direct': 3.114213197969538,
 'Bilirubin_total': 2.694403177550813,
 'Calcium': 8.316976957118822,
 'Chloride': 105.7650622558037,
 'Creatinine': 1.4043820374568778,
 'DBP': 59.98580935699335,
 'FiO2': 0.5262479604121526,
 'Fibrinogen': 292.25164179104473,
 'Gender': 0.5777212530766943,
 'Glucose': 133.6092206381394,
 'HCO3': 24.09447553326941,
 'HR': 84.98526444873023,
 'Hct': 30.67489522663289,
 'Hgb': 10.582028043138731,
 'ICULOS': 27.198518124814132,
 'Lactate': 2.4692027410382145,
 'MAP': 78.76734527184637,
 'Magnesium': 2.0410037247279824,
 'O2Sat': 97.26568772153938,
 'PTT': 40.78193651125141,
 'PaCO2': 41.16614709617827,
 'Phosphate': 3.588572789252058,
 'Platelets': 199.61784112312858,
 'Potassium': 4.161507176476068,
 'Resp': 18.773459507375623,
 'SBP': 120.96235945816058,
 'SaO2': 91.21545582226761,
 'Temp': 37.02673699236573,
 'TroponinI': 9.288186528497409,
 'WBC': 11.936603760868179,
 'pH': 7.380242785410664}
  all_columns=['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']
def fill_null_with_mean(data):
    data=pd.DataFrame(data,columns= all_columns)
    feature_name=[i for i in all_columns if i not in ["BaseExcess","Gender","EtCO2","Unit1","Unit2","HospAdmTime"]]
    for col in feature_name:
        if col=="SepsisLabel":
            continue
        else:
            """df[col].fillna(method="pad",inplace=True)#padding with before data"""
            df[col].fillna(AB_features_mean_dict[col],inplace=True) #if first is null fill mean data
    return np.array(df[feature_name][0:])
  
def get_sepsis_score(current_data,model):#current
    data=fill_null_with_mean(current_data)
    new_array=np.zeros((1,(data.shape[1])*5))
    array_data=np.array(data) 
    if data.shape[0]==1:
        new_array[0,0:data.shape[1]]= array_data[0].reshape(1,-1)
        new_array[0,data.shape[1]:data.shape[1]*2]=0
        new_array[0,data.shape[1]*2:] = list(array_data[0])*3
        add_data=np.array(new_array[0]).reshape(1,-1)
    else:
        trend_data=(np.diff(array_data.T))[:,-1].reshape(1,-1)   
        add_data=np.concatenate((array_data[-1].reshape(1,-1),trend_data,array_data.min(axis=0).reshape(1,-1),array_data.max(axis=0).reshape(1,-1),array_data.mean(axis=0).reshape(1,-1)),axis=1)
    print("add_data.shape:",add_data.shape)
    if add_data.shape[1]==170:
        score =model.predict_proba(add_data)[:,1]
        label = score > 0.4
    return score,label
    else:
        print("the shape is not match 170 ,check")
    return None
def load_sepsis_model():
    model =joblib.load("./train56000_model.m")
    return model

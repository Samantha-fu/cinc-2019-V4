import numpy as np
from sklearn.externals import joblib
def get_sepsis_score(data,model):#current 
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

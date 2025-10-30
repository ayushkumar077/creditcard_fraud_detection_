import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import streamlit as st
df=pd.read_csv('creditcard.csv')
legit=df[df.Class==0]
fraud=df[df.Class==1]
legit_sample=legit.sample(n=492)
new_data=pd.concat([legit_sample,fraud],axis=0)
X=new_data.iloc[:,0:-1]
y=new_data['Class']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
from sklearn.metrics import accuracy_score
accurcay=accuracy_score(y_pred,y_test)


#web app
st.title('Credit Card fraud Detection model')
input_df=st.text_input('Enter all required features values')
input_df_splited = input_df.split(',')


submit=st.button('submit')

if submit:
    features=np_df=np.asarray(input_df_splited,dtype=np.float64)
    prediction=model.predict(features.reshape(1,-1))

    if prediction[0]== 0 :
        st.write('Legitimate Transaction')
    else:
        st.write('Fradulant Transaction')
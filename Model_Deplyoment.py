#!/usr/bin/env python
# coding: utf-8

# In[6]:


import streamlit as st
import pickle
import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler


# In[7]:


log_model = pickle.load(open('Log_Model.pkl', 'rb'))
log_model
std_sca = pickle.load(open('Standard_scaler.pkl', 'rb'))
std_sca


# In[8]:


st.title("Model With Logistic Regression")
def user_input_parameters():
    CLMSEX = st.sidebar.radio('Select Your Gender',('Male','Female'))
    CLMINSUR = st.sidebar.selectbox('Enter Insurance Details,Yes-1,No-0',[1,0])
    SEATBELT= st.sidebar.selectbox('Enter Seatbelt Details,Yes-1,No-0',[1,0])
    CLMAGE = st.sidebar.slider('Select your age',1,100)
    LOSS = st.sidebar.number_input('Enter the loss')
    dict1={'CLMSEX':CLMSEX,'CLMINSUR':CLMINSUR,'SEATBELT':SEATBELT,'CLMAGE':CLMAGE,'LOSS':LOSS}
    features= pd.DataFrame(dict1,index=[0]) #index number to start from
    features['CLMSEX']=features['CLMSEX'].map({'Male': 1,'Female': 0})
    features[['CLMAGE','LOSS']]= std_sca.transform(features[['CLMAGE','LOSS']])
    return features 
df = user_input_parameters()
pred = log_model.predict(df) # This will give the output btw 0 & 1
pred_prob = log_model.predict_proba(df)# It will Provide 
# Probilit of 0 and Probabiltiy of 1
button = st.button('Predict')
if button is True:
    st.subheader('Prediction')
    st.write('Eligible' if pred_prob[0][1]>=0.5 else 'Not Eligible')
    st.subheader('Predicted Prob')
    st.write(pred_prob)


# So to Creating the things we should focus on and in what order we need to write code.
# * Keep the order of the features(input) as we trained our Logistic regression Model.
# 
# #### The Code Explantion
# * Here we Created a funtion with the name of the user_input.
# * Herer we will get the input from the user and store them and converted it to Data Frame using the pandas function.
# * Mapped to Male=1 and Female=0 using the manual mapping with pandas map function.
# * Applied the scaling on the CLMAGE and LOSS so user entered output should get scaled in backend before the data passed to the model.
# * Then used return the features. Means when we use "user_input" function it will take the user input what process we did in the features (like we applied mapping scaling) it will show that output in the DATA_FRAME foramt.
# * stored user_input function in a varible. the user_input is taken as testing based on the input provided it will show us prediction.
# * The logstic regression will give the output in the 0 & 1 based on the sigmoid( "if>=0.5=>1" if<0.5=>0"). So to show the probability of 1 & 0
# * So we used the button option when pressed on that button it will give the heading Prediction. if probabiltiy is >=0.5 it will give "eligible". if <0.5 it will give "not_eligble".
# * 

# In[ ]:





import joblib

def predict(data):
    clf = joblib.load("rf_model.sav")
    return clf.predict(data)

def predict_proba(data):
    clf = joblib.load("rf_model.sav")
    return clf.predict_proba(data)

import streamlit as st
import pandas as pd
import numpy as np
#from prediction import predict


st.title('Identyfing VIP players in first 10 days after sign up :trophy:')
#st.markdown('This model aims to detect VIP players \
#      based on their performance in first 10 days after registration \
#    .')

#st.subheader("Players performance in first 10 days after registration")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Monetary characteristics")
    dep_amt = st.slider('dep_amt', 0, 10000, 1500)
    paid_bonus = st.slider('paid_bonus', 0, 1000, 100)
    stake_amt = st.slider('stake_amt', 0, 50000, 15000)
    wth_amt = st.slider('wth_amt', 0, 10000, 300)


with col2:
    st.subheader("Engagement characteristics")
    dep_days_greater100e = st.slider('dep_days_greater100e', 0, 10, 6)
    wth_days_greatter100e = st.slider('wth_days_greatter100e', 0, 10, 1)
    abs_ggr_days_greater200e = st.slider('abs_ggr_days_greater200e', 0,10,3)
    no_logins = st.slider('no_logins', 0, 50, 10)    
    stake_days_greater500e = st.slider('stake_days_greater500e', 0, 10, 4)
    no_bets = st.slider('no_bets', 0, 10000, 100) 
st.text('')
if st.button("Predict type of player"):
    result = predict(
        np.array([[dep_amt,dep_days_greater100e,wth_amt,wth_days_greatter100e,stake_amt,stake_days_greater500e,paid_bonus,no_bets,abs_ggr_days_greater200e,no_logins]]))
    st.text(result[0])


from sklearn.ensemble import RandomForestClassifier

result_proba = predict_proba(np.array([[dep_amt,dep_days_greater100e,wth_amt,wth_days_greatter100e,stake_amt,stake_days_greater500e,paid_bonus,no_bets,abs_ggr_days_greater200e,no_logins]]))
if result_proba[0,0]>0.5:
    st.subheader('This player is likely to be VIP:thumbsup:')
else:
    st.subheader('This player is NOT likley to be VIP :thumbsdown:')
st.write(result_proba)
#st.text('')
#if st.button("Predict type of Iris"):
 #   result = predict(
  #      np.array([[sepal_l, sepal_w, petal_l, petal_w]]))
   # st.text(result[0])
st.text('')
st.text('')
st.markdown(
    '`Created by` Fincore')


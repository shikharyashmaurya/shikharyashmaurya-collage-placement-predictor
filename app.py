import pandas as pd

# new_data = pd.DataFrame({
#     'gender':0,
#     'ssc_p':55.0,
#     'ssc_b':0,
#     'hsc_p':75.0,
#     'hsc_b':0,
#     'hsc_s':1,
#     'degree_p':65.0,
#     'degree_t':2,
#     'workex':0,
#     'etest_p':55.0,
#      'specialisation':1,
#     'mba_p':58.8,
# },index=[0])

import joblib

model = joblib.load('model_campus_placement')

# p=model.predict(new_data)
# prob=model.predict_proba(new_data)

# if p[0]==1:
#     print('Placed')
#     print(f"You will be placed with probability of {prob[0][1]:.2f}")
# else:
#     print("Not-placed")

import streamlit as st

st.title('Campus-Placement-Prediction-Using-Machine-Learning')

gender=st.number_input('gender',value=0)
ssc_p=st.number_input('ssc_p',value=0)
ssc_b=st.number_input('ssc_b',value=0)
hsc_p=st.number_input('hsc_p',value=0)
hsc_b=st.number_input('hsc_b',value=0)
hsc_s=st.number_input('hsc_s',value=0)
degree_p=st.number_input('degree_p',value=0)
degree_t=st.number_input('degree_t',value=0)
workex=st.number_input('workex',value=0)
etest_p=st.number_input('etest_p',value=0)
specialisation=st.number_input('specialisation',value=0)
mba_p=st.number_input('mba_p',value=0)

if st.button('submit'):

    new_data = pd.DataFrame({
        'gender':gender,
        'ssc_p':ssc_p,
        'ssc_b':ssc_b,
        'hsc_p':hsc_p,
        'hsc_b':hsc_b,
        'hsc_s':hsc_s,
        'degree_p':degree_p,
        'degree_t':degree_t,
        'workex':workex,
        'etest_p':etest_p,
        'specialisation':specialisation,
        'mba_p':mba_p,
    },index=[0])

    p=model.predict(new_data)
    prob=model.predict_proba(new_data)

    
    if p[0]==1:
        # print('Placed')
        st.write('Placed')
        # print(f"You will be placed with probability of {prob[0][1]:.2f}")
        st.write(f"You will be placed with probability of {prob[0][1]:.2f}")

    else:
        # print("Not-placed")
        st.write('Not-placed')












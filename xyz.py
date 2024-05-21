import streamlit as st 
import joblib
import pandas as pd
#st.title('Placement Prediction app')
# st.markdown("""
#     <style>
#     .title {
#         font-size: 50px;
#         font-weight: bold;
#         color: #4CAF50;
#         text-align: center;
#         font-family: 'Courier New', Courier, monospace;
#     }
#     </style>
#     """, unsafe_allow_html=True)


# st.title('Placement Prediction App')
# st.subheader('Predicting student placement outcomes using machine learning')
# st.markdown('This app uses historical data to predict whether a student will be placed in a company based on their profile.')

try:
    model = joblib.load('model_campus')
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()
# model = joblib.load(open('model_campus_placement_rf.joblib','rb'))

def predict_placement(data):
    # Preprocess the data
    # new_data = pd.DataFrame(data)
    new_data = pd.DataFrame(data, index=[0])

    # Make prediction
    prediction = model.predict(new_data)[0]
    prob = model.predict_proba(new_data)[0][1]

    return prediction, prob

def main():
    st.header('Placement Prediciton App')

    gender = st.radio('Gender', ['Male', 'Female'])
    ssc_p = st.number_input('Secondary School Percentage', min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    ssc_b = st.radio('Board of Education (SSC)', ['Central', 'Others'])
    hsc_p = st.number_input('Higher Secondary Percentage', min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    hsc_b = st.radio('Board of Education (HSC)', ['Central', 'Others'])
    degree_p = st.number_input('UG Percentage', min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    branch = st.selectbox('Branch of Study', ['CSE', 'ECE/EN', 'Others'])
    workex = st.radio('Work Experience', ['Yes', 'No'])
    certifications = st.number_input('Number of Certifications', min_value=0, max_value=10, value=0)
    etest_p = st.number_input('Employability Percentage', min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    backlogs = st.number_input('Number of Backlogs', min_value=0, max_value=10, value=0)
    
    if st.button('predict'):
        new_data = {
            'gender': 0 if gender == "Male" else 1,
            'ssc_p': ssc_p,
            'ssc_b': 1 if ssc_b == "Central" else 0,
            'hsc_p': hsc_p,
            'hsc_b': 1 if hsc_b == "Central" else 0,
            'degree_p': degree_p,
            'Branch': 2 if branch == "ECE/EN" else 1 if branch == "CSE" else 0,
            'Workex': 1 if workex == "Yes" else 0,
            'Certifications': certifications,
            'etest_p': etest_p,
            'Backlogs': backlogs,
    }
        
        # st.write(new_data)

        prediction, probability = predict_placement(new_data)
        st.write(f'Percentage of getting placed: {probability*100:.2f}%')


if __name__=='__main__':
    main()


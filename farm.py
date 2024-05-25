import streamlit as st
import numpy as np 
import pandas as pd
import pickle

st.title("Crop Price Prediction")

df = pickle.load(open('df.pkl','rb'))
pipe = pickle.load(open('pipe.pkl','rb'),encoding='latin1')

X = df.drop('Price_per_KG_INR',axis=1)
y = df.iloc[:,-1]

state = st.selectbox('Select State:',df['State'].unique())

dist = st.selectbox('Select District:',df['District'].unique())

year = st.selectbox('Select Year:',df['Crop_Year'].unique())

season = st.selectbox('Season:',df['Season'].unique())

crop = st.selectbox('Select Crop:',df['Crop'].unique())

Area = st.number_input(label='Total Area:',step=1,format='%.2d',min_value=0,max_value=1250)


Production = st.number_input(label='Total Production:',step=1,format='%.2d',min_value=0,max_value=1250)

btn = st.button('Predict Price')

if btn:
    query = pd.DataFrame(data=[[state,dist,year,season,crop,Area,Production]],columns=X.columns)
    st.title('Predicted Price is:' +str(round(pipe.predict(query)[0],2)))

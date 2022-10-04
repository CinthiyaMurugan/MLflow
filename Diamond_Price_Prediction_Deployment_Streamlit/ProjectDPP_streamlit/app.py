import streamlit as st 
import numpy as np
import pandas as pd
from pickle import load
from PIL import Image



c1,c2, = st.columns(2)
with c1:
    st.title("Diamond Price Prediction ")
with c2:
    image = Image.open('img/th.jpg')
    st.image(image)

scaler=load(open('models/standard_scaler.pkl','rb'))
rf_model=load(open('models/randomforest_model.pkl','rb'))

clarity_encoder={'I1':1,'SI2':2,'SI1':3,'VS2':4,'VS1':5,'VVS2':6,'VVS1':7,'IF':8}
color_encoder={'J':1,'I':2,'H':3,'G':4,'F':5,'E':6,'D':7}
cut_encoder={'Fair':1,'Good':2,'Very Good':3,'Ideal': 4,'Premium':5}

column1, column2, column3 = st.columns(3)

with column1:
    carat=st.text_input('Carat',placeholder='Enter value in mm (range 0.2-5.01)')
    depth=st.text_input('Depth',placeholder='Enter value in mm (range 43-79)')
    table=st.text_input('Table',placeholder='Enter value in mm (range 43-95)')
    

with column2:
    cut = st.selectbox(
        'How would be the cut of Diamond?',
        ('select option','Fair', 'Good', 'Very Good', 'Ideal', 'Premium'))

    color = st.selectbox(
        'What should be the color of Diamond?',
        ('select option','J', 'I', 'H', 'G', 'F', 'E', 'D'))

    clarity = st.selectbox(
        'How would you like to be contacted?',
        ('select option','I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'))


with column3:
    btn_click=st.button("Click to Predict the Diamond Price in USD")

    if btn_click==True:
        if carat and depth and table:
        
        
            query_point_num_transformed=scaler.transform([[float(carat),float(depth),float(table)]])
            query_point_cat=np.array([clarity_encoder[clarity],color_encoder[color],cut_encoder[cut]])#.reshape(1,-1)
        
            df=np.concatenate((query_point_cat,query_point_num_transformed.flatten()),axis=None)
            pred=rf_model.predict(df.reshape(1,-1)).item()
            st.success(pred)
            st.balloons()
        else:
            st.error('Enter the values properly')
            st.warning()
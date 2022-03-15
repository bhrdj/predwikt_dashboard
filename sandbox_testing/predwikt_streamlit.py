#!/usr/bin/env python

import streamlit as st
import seaborn as sns 
import matplotlib.pyplot as plt 
import pandas as pd 
import pickle 
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import GridSearchCV

st.title('Predwikt')

datapath = './data/'
with open(datapath + 'disaster_daily_edits.pickle', 'rb') as f:
    disaster_daily_edits = pickle.load(f)

train_weeks = st.slider('min_train', 
                        min_value=0, 
                        max_value=disaster_daily_edits.shape[0]-200)
start_date = st.slider('max_train', min_value=min_train+100, max_value=disaster_daily_edits.shape[0]-2)


st.write('hiiiii')
#!/usr/bin/env python

import streamlit as st
import seaborn as sns 
import matplotlib.pyplot as plt 
import pandas as pd 
import pickle 
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import GridSearchCV

st.title('Predwikt')

datapath = '../data/'
with open(datapath + 'disaster_daily_edits.pickle', 'rb') as f:
    disaster_daily_edits = pickle.load(f)

st.

st.write('hiiiii')
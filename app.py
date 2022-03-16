import pandas as pd
import numpy as np
import datetime
from holidays_jp import CountryHolidays
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import pickle
import streamlit as st
#  import altair as alt
import matplotlib.pyplot as plt
import os

@st.cache
def load_data():
    # get data in memory
    infile = open('assets/disaster_daily_edits.pickle','rb') 
    daily_edits = pickle.load(infile)
    infile.close()

    # get japanese holidays in date range
    holidays2019 = CountryHolidays.get('JP', 2019)
    holidays2020 = CountryHolidays.get('JP', 2020)
    holidays2021 = CountryHolidays.get('JP', 2021)
    holiday_list = [holidays2019, holidays2020, holidays2021]
    holidays = pd.concat(map(pd.DataFrame, holiday_list), axis='rows').set_index(0)        
    holidays.index = holidays.index.tz_localize('Japan')
    holidays = holidays.resample('D').asfreq(' ').rename(columns={1:'holiday_name'})
    holidays['holiday'] = holidays.holiday_name.map(lambda x: int(x != ' '))

    # prep data
    disasters_english = {'火山災害':'VolcanicDisaster', 
                        '熱帯低気圧':'TropicalCyclones', 
                        '雪害':'SnowDamage', 
                        '地震':'Earthquake', 
                        '津波':'Tsunami'}       
    ts = daily_edits.copy()
    ts = ts.rename(columns=disasters_english)
    ts['mtwtf'] = ts.index.dayofweek.isin([0,1,2,3,4]).astype(int)
    ts['sat'] = ts.index.dayofweek.isin([5]).astype(int)
    ts['sun'] = ts.index.dayofweek.isin([6]).astype(int)
    ts['holiday'] = holidays['holiday']
    ts['holiday_on_weekday'] = ts[['holiday', 'mtwtf']].all(axis='columns').astype(int)
    ts = ts[ts.columns.difference(['holiday'])]
    ts = ts.iloc[1:-1]    
    calendar_cols = ['holiday_on_weekday','mtwtf','sat', 'sun',]
    return ts, list(disasters_english.values()), calendar_cols

# @st.cache
# def instantiate_results(counter):
#     results = []
#     return results

def model_prep(target_names,ts,p_AR_parameter,moving_average):
    ts_lags = ts.copy()

    lag_vars, μ_vars = [], []
    for i in target_names:
        for j in list(range(1,p_AR_parameter+1)):
            ts_lags[f"{i}_l{j}"] = ts_lags[i].shift(j)
            lag_vars.append(f"{i}_l{j}")
        for j in [moving_average]:
            ts_lags[f"{i}_μ{j}"] = ts_lags[i].rolling(window=j, closed="left").mean()
            μ_vars.append(f"{i}_μ{j}")
            
    ts_lags = ts_lags.dropna()

    XX = ts_lags[ts_lags.columns.difference(target_names)]
    YY = ts_lags[target_names]

    return XX, YY

def feature_columns(XX,target_names,calendar_cols):
    Xcols = {}
    for i in target_names:
        Xcols[i] = XX.columns[XX.columns.str.startswith(i)].tolist() + calendar_cols
    return Xcols

@st.cache
def model_fit(ts,p_AR_parameter,moving_average,target_names,calendar_cols):
    XX,YY = model_prep(target_names,ts,p_AR_parameter,moving_average)

    start_tr = 0
    end_tr = 730
    end_vl = XX.shape[0]
    XXtr, YYtr = XX.copy().iloc[start_tr:end_tr], YY.iloc[start_tr:end_tr]
    XXvl, YYvl = XX.copy().iloc[end_tr:end_vl], YY.iloc[end_tr:end_vl]

    Xcols = feature_columns(XX,target_names,calendar_cols)

    ycols = {i:i for i in target_names}

    Xtr, Xvl = {}, {}
    ytr, yvl = {}, {}
    for diz in target_names:
        Xtr[diz] = XXtr[Xcols[diz]]
        Xvl[diz] = XXvl[Xcols[diz]]
        ytr[diz] = YYtr[ycols[diz]]
        yvl[diz] = YYvl[ycols[diz]]

    ri, gs_ri, scores = {}, {}, {}
    for diz in Xtr:
        ri[diz] = Ridge()
        gs_ri[diz] = GridSearchCV(ri[diz], {'alpha': [-100,-10,0,10,100]})
        gs_ri[diz].fit(Xtr[diz],ytr[diz])
        scores[diz] = gs_ri[diz].score(Xvl[diz],yvl[diz])
        
    return gs_ri,scores,XX,YY,Xcols

def grid_search():
    # load data - should only happen once
    ts, target_names_default, calendar_cols = load_data()
    st.title("Manual Grid Search")
    # user input
    with st.form("inputs"):

        target_names = st.multiselect(
            "Select your target",
            target_names_default,
            default = target_names_default
        )
        p_AR_parameter = st.select_slider("select # of lags",
        options=range(1,6),value=3
        )
        moving_average = st.select_slider(
            "select # of days for moving average",
            options=range(7,14)
        )
        
        submitted = st.form_submit_button("Compute!")
    
    gs_ri,scores,XX,YY,Xcols = model_fit(ts,p_AR_parameter,moving_average,target_names,calendar_cols)
    
    result_dict = {}
    result_dict['target_names'] = target_names
    result_dict['p_AR_parameter'] = p_AR_parameter
    result_dict['moving_average'] = moving_average

    # for diz in target_names:
    #     result_dict[diz+'_score'] = scores[diz]
    #     st.write(f"{diz}: {scores[diz]:.2f}")
    round_scores = {i:round(scores[i], 3) for i in scores}
    st.write("Results: Quality of Fit")
    st.dataframe(pd.DataFrame(scores, index=['R²']))
    st.write("Results: Model Coefficients")
    st.dataframe(get_coeffs(gs_ri,target_names,p_AR_parameter,moving_average))
    st.write("Actual and Predicted Wikipedia Edits by Category")
    model_plot(result_dict, gs_ri,target_names,XX,YY,Xcols)
    
def get_coeffs(gs_ri,target_names,p_AR_parameter,moving_average):
    coeffs = {}
    AR_coeff_nums = list(range(1,p_AR_parameter+1))
    AR_coeff_names = list(['lag_' + str(i) + '_days' for i in AR_coeff_nums])
    average_coeff_name = ['μ'+str(moving_average)]
    coeff_names = AR_coeff_names + average_coeff_name + ['holiday_on_weekday', 'mtwtf', 'sat', 'sun']
    
    for diz in gs_ri:
        coeffs[diz] = pd.Series(
            gs_ri[diz].best_estimator_.coef_, 
            index=coeff_names, 
            name=diz)
    return pd.DataFrame(coeffs)
    
def model_plot(result_dict, gs_ri,target_names,XX,YY,Xcols):
    diz_colors = {'VolcanicDisaster':'red', 
                  'TropicalCyclones':'blue', 
                  'SnowDamage':'black', 
                  'Earthquake':'green', 
                  'Tsunami':'purple'}
    diz_colors_light = {'VolcanicDisaster':'pink', 
                  'TropicalCyclones':'lightblue', 
                  'SnowDamage':'gray', 
                  'Earthquake':'lightgreen', 
                  'Tsunami':'violet'}
    fig, axes = plt.subplots(nrows=len(target_names), ncols= 1, 
                             figsize=(8,8), sharex=True)
    for diz,ax in zip(target_names,axes):
        YY_pred = gs_ri[diz].best_estimator_.predict(XX[Xcols[diz]])
        YY_pred = pd.Series(YY_pred, index=XX[Xcols[diz]].index, name="prediction")
        actual, = ax.plot(YY[diz], color='gray', linestyle='dashed', alpha = .7)
        predicted, = ax.plot(YY_pred, color=diz_colors[diz], alpha = 1)
        ax.set_ylabel(diz)
        ax.yaxis.label.set_color(diz_colors[diz])
        ax.legend([actual, predicted], ['actual', 'predicted'])
    plt.xticks(rotation = 90)
    st.pyplot(fig)

def main():
    page = st.sidebar.selectbox('Choose your page',['Home','GridSearch'])
    if page == 'Home':
        st.title("Forecasting Wikipedia Edits")
        st.markdown("""
        Capstone Project  
        Data Science 2021-2022  
        General Assembly  
        **Steven Bhardwaj**
        """)
    else:
        import os
        grid_search()

if __name__ == "__main__":
    main()

import pickle
import pandas as pd
import numpy as np
import streamlit as st
import datetime
import altair as alt
from holidays_jp import CountryHolidays
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

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
    ts['sun'] = ts.index.dayofweek.isin([5]).astype(int)
    ts['holiday'] = holidays['holiday']
    ts['holiday_on_weekday'] = ts[['holiday', 'mtwtf']].all(axis='columns').astype(int)
    ts = ts[ts.columns.difference(['holiday'])]
    ts = ts.iloc[1:-1]    
    calendar_cols = ['holiday_on_weekday','mtwtf','sat', 'sun',]
    return ts, list(disasters_english.values()), calendar_cols



# def chart_data(data,disasters):
#     #filter
#     data_prep = data.loc[:,disasters].copy()

#     #transform
#     data_prep.reset_index(inplace=True) # get the column as a selectable field, named 'day'
#     source = pd.melt(data_prep,id_vars='day',value_vars=disasters,var_name='disaster',value_name='edits')

#     # plot
#     # https://altair-viz.github.io/gallery/multiline_tooltip.html
#     nearest = alt.selection(type='single', nearest=True, on='mouseover',
#                         fields=['day'], empty='none')

#     # The basic line
#     line = alt.Chart(source).mark_line().encode(
#         x='day',
#         y='edits',
#         color='disaster'

#     )

#     # Transparent selectors across the chart. This is what tells us
#     # the x-value of the cursor
#     selectors = alt.Chart(source).mark_point().encode(
#         x='day:Q',
#         opacity=alt.value(0),
#     ).add_selection(
#         nearest
#     )

#     # Draw points on the line, and highlight based on selection
#     points = line.mark_point().encode(
#         opacity=alt.condition(nearest, alt.value(1), alt.value(0))
#     )

#     # Draw text labels near the points, and highlight based on selection
#     text = line.mark_text(align='left', dx=5, dy=-5).encode(
#         text=alt.condition(nearest, 'edits:Q', alt.value(' '))
#     )

#     # Draw a rule at the location of the selection
#     rules = alt.Chart(source).mark_rule(color='gray').encode(
#         x='day:Q',
#     ).transform_filter(
#         nearest
#     )

#     # Put the five layers into a chart and bind the data
#     chart = alt.layer(
#         line, selectors, points, rules, text
#     ).properties(
#         width=600, height=300
#     )

#     return chart

@st.cache
def model_fit(ts,p_AR_parameter,moving_average,target_names,calendar_cols):
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

    start_tr = 0
    end_tr = 600
    end_vl = 800
    XXtr, YYtr = XX.copy().iloc[start_tr:end_tr], YY.iloc[start_tr:end_tr]
    XXvl, YYvl = XX.copy().iloc[end_tr:end_vl], YY.iloc[end_tr:end_vl]

    Xcols = {}
    for i in target_names:
        Xcols[i] = XX.columns[XX.columns.str.startswith(i)].tolist() + calendar_cols

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

        # print((diz, gs_ri[diz].score(Xvl[diz],yvl[diz]), gs_ri[diz].best_params_))
        
    return gs_ri,scores,XX,YY,Xcols

def model_plot(gs_ri,XX,YY,Xcols,target_names):
    for diz in target_names:
        YY_pred = gs_ri[diz].best_estimator_.pred(XX[Xcols])

def main():
    # load data - should only happen once
    ts, target_names_default, calendar_cols = load_data()
    st.title("Define Model")
    # user input
    with st.form("inputs"):
        # start_date = st.date_input(
        #     "Start Date of Interest",
        #     datetime.date(2019,1,1)
        # )
        # end_date = st.date_input(
        #     "End Date of Interest",
        #     datetime.date(2021,1,1)
        # )
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
    
    for diz in target_names:
        st.write(f"{diz}: {scores[diz]}")
    


    # produce a chart
    #st.write(chart_data(data,disasters).head())
    # st.altair_chart(chart_data(data,disasters))



if __name__ == "__main__":
    main()

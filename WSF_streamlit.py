#Imports
import streamlit as st
import pandas as pd
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle, islice
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from matplotlib.ticker import MaxNLocator
from datetime import datetime
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
import bz2
import pickle
import _pickle as cPickle

#stop Warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

# Funtion to read the Data
def read_data(filename):
    """ A funtion to read the pickle file"""
    df = pd.read_pickle(filename+".pkl")
    time.sleep(8)
    st.write("Data read successfully")
    st.write(df.shape)
    return df
    
# Function find Sales decomposition
def sales_analysis(feat,param, resample_method, decomp, master_df1):
    '''
    This function applies a seasonal decomposition to a time series. It will generate a season plot, a trending plot, and, finally, a resid plot

    Args:
       feat : Feature is the feature column value  which is state, category, department or store id
       param: Param is sub-category of a feature
       resample_method: Seasonal decomposition requires resample methods
       decomp: Decompose sales or not?
       master_df1: Dataset
    '''
    st.write(master_df1.shape)
    sales_df = master_df1.loc[master_df1[feat] == param]
    sales_df['date'] = pd.to_datetime(sales_df['date'])
    sales_df =sales_df.groupby('date')['sales'].sum().reset_index()
    sales_df = sales_df.set_index('date')
    resample_df = sales_df['sales'].resample(resample_method).mean()
    colors = ["blue","black","brown","red","yellow","green","orange","turquoise","magenta","cyan"]
    random.shuffle(colors)
    
    fig = plt.Figure(figsize=(12,7))
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312)
    ax3 = plt.subplot(313)

    decomposition = sm.tsa.seasonal_decompose(resample_df)
    decomposition.seasonal.plot(color='green', ax=ax1, title='Seasonality')
    plt.legend('')
    
    decomposition.trend.plot(color='green', ax=ax2, title='Trending')
    plt.legend('')
 
    decomposition.resid.plot(color='green', ax=ax3, title='Resid')
    plt.legend('')
    plt.subplots_adjust(hspace=1)
    st.pyplot() 

    
# Modeling - item selection
def select_item(master_df, item_id):
    """Funtion to Select item for forecasting
    Args:
    Master_df: Dataset 
    product_id: the item id user wants to forecast for.
    """
    prod = master_df[master_df['id'] == item_id].reset_index()
    return prod


def autoarima(prod):
    """ A function to forecast using AutoARIMA model with zero differencing
    
    Args:
    prod: the item id user wants to forecast for.
    """
    
    train = prod['sales'][:1850]
    test = prod['sales'][1850:]
    smodel = pm.auto_arima(train,start_p=1, start_q=1,
                             test='adf',
                             max_p=3, max_q=3, m=12,
                             start_P=0, seasonal=True,
                             d=None, D=0, trace=True,
                             error_action='ignore',  
                             suppress_warnings=True, 
                             stepwise=True)

    st.write(smodel.summary())
    
    # Forecast
    n_periods = 62
    fitted, confint = smodel.predict(n_periods=n_periods, return_conf_int=True)

    # make series for plotting purpose 
    fitted_series = pd.Series(fitted, index= range(1850,1912,1))
    lower_series = pd.Series(confint[:, 0],index=range(1850,1912,1))
    upper_series = pd.Series(confint[:, 1],index=range(1850,1912,1))
    rmse = np.mean((fitted_series - test)**2)**.5
    mae = np.mean(np.abs(fitted_series - test))
    st.write("Model's RMSE: ", rmse,"Model's MAE: ", mae)

    # Plot
    plt.figure(figsize=(12,5), dpi=100)
    plt.plot(train[1700:], label = "training")
    plt.plot(test, label = "test")
    plt.plot(fitted_series, label = "forecast")
    plt.fill_between(lower_series.index, 
                     lower_series, 
                     upper_series, 
                     color='k', alpha=.05)
    plt.title('Forecast vs Actuals for item {}'.format(item))
    plt.xlabel("days")
    plt.ylabel("Sales of the item")
    plt.legend()
    plt.show()
    st.pyplot(fig)

    
def autoarima_differenced(prod):
    train = prod['sales'][:1850]
    smodel = pm.auto_arima(train,start_p=1, start_q=1,
                             test='adf',
                             max_p=4, max_q=4, m=12,
                             start_P=0, seasonal=True,
                             d=None, D=1, trace=True,
                             error_action='ignore',  
                             suppress_warnings=True, 
                             stepwise=True, n_jobs = -1)
    return smodel

def sarimax_model_forecast(prod, order, seasonal_order,seasonal_variables, item):
    exog_train = prod[seasonal_variables].iloc[:1850, :]
    exog_val = prod[seasonal_variables].iloc[1850:1913, :]
    exog_test = prod[seasonal_variables].iloc[1850:1943, :]
    
    mod = sm.tsa.statespace.SARIMAX(prod.sales[:1850],trend='n', exog=exog_train, order= order, seasonal_order=seasonal_order)
    results = mod.fit() 
    forecast_val = pd.DataFrame(results.predict(start = 1850, end = 1912, exog=exog_val))
    forecast_val['predicted_mean'] = np.where((forecast_val.predicted_mean < 0),0,round(forecast_val.predicted_mean))
    forecast_test = pd.DataFrame(results.predict(start = 1912, end = 1940, exog=exog_test))
    forecast_test['predicted_mean'] = np.where((forecast_test.predicted_mean < 0),0,round(forecast_test.predicted_mean))
    
    rmse = np.mean((forecast_val['predicted_mean'] - prod.sales[1850:])**2)**.5
    #mae = np.mean(np.abs(forecast_val['predicted_mean']) - prod.sales[1850:])
    print("Model's RMSE: ", rmse)
    plt.figure(figsize= (18,6))
    #results.summary()
    plt.plot(prod.sales[1850:], label = "actual")
    plt.plot(forecast_val['predicted_mean'], label = 'Predicted')
    plt.title('Forecast vs Actuals for item {} using SARIMA Model'.format(item))
    plt.xlabel("days")
    plt.ylabel("Sales of the item")
    plt.legend()
    plt.show()
    return rmse, forecast_test

def plot_forecast(forecast, item):
    plt.figure(figsize = (12, 6))
    plt.plot(forecast, marker='o', color='b')
    plt.xticks(rotation = 90)
    plt.title('28 days forecast for item {} using SARIMA Model'.format(item))
    plt.xlabel("days")
    plt.ylabel("Sales of the item")
    plt.legend(["forecast"])
    plt.show()
    st.pyplot()

    
# Load any compressed pickle file
def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data


def main():
    
    """Main function with all the radio button actions"""
    
    st.image("walmartlogo.PNG", use_column_width  = "auto")
    st.title('Walmart Sales Forecasting')

    # Add a selectbox to the sidebar:
    activities = ["Data Analysis","Data Visualization", "Modeling", "About"]
    choice = st.sidebar.radio("Select from the Following: ",activities)
    
    
    calendar = pd.read_csv("calendar.csv")

    if choice == 'Data Analysis':
        st.header("Data Analysis")

        #Select Feature
        
        feat = st.selectbox('Select parameter you want to decompose sales for:',('state_id', 'cat_id', 'dept_id', 'store_id'))

        st.write('You selected:', feat)
        st.write('You select the type of feature')


        #Select feature
        if feat == "state_id":
                train_val_df = decompress_pickle('train_val_df.pbz2')
                if st.button("Generate plot for state_id"):
                    temp1 = train_val_df.groupby(['state_id'])['store_id'].count()
                    plt.figure(figsize=(10,6))
                    temp1.plot(kind='bar',color=['r', 'g', 'b'])
                    plt.title("Number of sales by State")
                    st.pyplot()
                    
                    
                param = st.selectbox('Select parameter you want to decompose sales for:',('CA', 'TX', 'WI'))

                if param == "CA":
                    if st.button("Generate analysis for CA"):
                        #temp2 = train_val_df.groupby(['store_id']).sum()
                        fig,axes=plt.subplots(1,1, figsize=(20, 8))
                        plt.tick_params(labelsize=14)
                        plt.plot(train_val_df.groupby(['store_id']).sum().loc['CA_1'])
                        plt.plot(train_val_df.groupby(['store_id']).sum().loc['CA_2'])
                        plt.plot(train_val_df.groupby(['store_id']).sum().loc['CA_3'])
                        plt.plot(train_val_df.groupby(['store_id']).sum().loc['CA_4'])

                        plt.legend(train_val_df.groupby(['store_id']).sum().index)
                        axes.xaxis.set_major_locator(MaxNLocator(15)) 
                        axes.yaxis.set_major_locator(MaxNLocator(15))
                        axes.set_title("Number of sales by stores in california each day", fontsize = 18)
                        axes.set_xlabel("day", fontsize = 18)
                        axes.set_ylabel("Number of Sales", fontsize = 18)
                        plt.show()
                        st.pyplot()


                if param == "TX":
                    if st.button("Generate analysis for TX"):
                        fig,axes=plt.subplots(1,1, figsize=(20, 8))
                        plt.tick_params(labelsize=14)
                        plt.plot(train_val_df.groupby(['store_id']).sum().loc['TX_1'])
                        plt.plot(train_val_df.groupby(['store_id']).sum().loc['TX_2'])
                        plt.plot(train_val_df.groupby(['store_id']).sum().loc['TX_3'])
                        plt.legend(['TX_1','TX_2','TX_3'])
                        axes.xaxis.set_major_locator(MaxNLocator(15)) 
                        axes.yaxis.set_major_locator(MaxNLocator(15))
                        axes.set_title("Number of sales by Stores ID in Texas", fontsize = 18)
                        axes.set_xlabel("day", fontsize = 18)
                        axes.set_ylabel("Stores sales in Texas", fontsize = 18)
                        plt.show()
                        st.pyplot()

                if param == "WI":
                     if st.button("Generate analysis for WI"):
                        fig,axes=plt.subplots(1,1, figsize=(20, 8))
                        plt.tick_params(labelsize=14)
                        plt.plot(train_val_df.groupby(['store_id']).sum().loc['WI_1'])
                        plt.plot(train_val_df.groupby(['store_id']).sum().loc['WI_2'])
                        plt.plot(train_val_df.groupby(['store_id']).sum().loc['WI_3'])
                        plt.legend(['WI_1','WI_2','WI_3'])
                        axes.xaxis.set_major_locator(MaxNLocator(15)) 
                        axes.set_title("Number of sales by Stores ID in Wisconsin each say", fontsize = 18)
                        axes.set_xlabel("day", fontsize = 18)
                        axes.set_ylabel("Number of sales in Wisconsin stores", fontsize = 18)
                        plt.show()
                        st.pyplot()


        elif feat == "cat_id":
                train_val_df = decompress_pickle('train_val_df.pbz2')
                if st.button("Generate plot for category"):
                    temp3 = train_val_df.groupby(['cat_id'])['dept_id'].count()
                    plt.figure(figsize=(10,6))
                    temp3.plot(kind='bar',color=['b', 'r', 'g'])
                    plt.title("Number of sales within Categories")
                    st.pyplot() 
                    
                    
                param = st.selectbox('Select parameter you want to decompose sales for:',('HOBBIES', 'FOODS', 'HOUSEHOLD'))
                if param == "HOBBIES":
                    if st.button("Generate analysis by HOBBIES"):
                        fig,axes=plt.subplots(1,1, figsize=(20, 8))
                        plt.tick_params(labelsize=14)
                        plt.plot(train_val_df.groupby(['dept_id']).sum().loc['HOBBIES_1'])
                        plt.plot(train_val_df.groupby(['dept_id']).sum().loc['HOBBIES_2'])
                        plt.legend(['HOBBIES_1','HOBBIES_2'])
                        axes.xaxis.set_major_locator(MaxNLocator(15)) 
                        axes.yaxis.set_major_locator(MaxNLocator(15))
                        axes.set_title("Number of sales by department Hobbies per day",  fontsize= 18)
                        axes.set_xlabel("Days",  fontsize= 18)
                        axes.set_ylabel("Number of Sales by department hobbies",  fontsize= 18)
                        plt.show()
                        st.pyplot()



                if param == "FOODS":
                    if st.button("Generate analysis by FOOD"):
                        fig,axes=plt.subplots(1,1, figsize=(18, 8))
                        plt.tick_params(labelsize=14)
                        plt.plot(train_val_df.groupby(['dept_id']).sum().loc['FOODS_1'])
                        plt.plot(train_val_df.groupby(['dept_id']).sum().loc['FOODS_2'])
                        plt.plot(train_val_df.groupby(['dept_id']).sum().loc['FOODS_3'])

                        plt.legend(train_val_df.groupby(['dept_id']).sum().index)
                        axes.xaxis.set_major_locator(MaxNLocator(15)) 
                        axes.yaxis.set_major_locator(MaxNLocator(15))
                        axes.set_title("Number of sales by food department each day", fontsize= 18)
                        axes.set_xlabel("Days",  fontsize= 18)
                        axes.set_ylabel("Number of Sales for Food department",  fontsize= 18)
                        plt.show()
                        st.pyplot()

                if param == "HOUSEHOLD":
                    if st.button("Generate analysis by HOUSEHOLD"):
                        fig,axes=plt.subplots(1,1, figsize=(21, 8))
                        plt.tick_params(labelsize=14)
                        plt.plot(train_val_df.groupby(['dept_id']).sum().loc['HOUSEHOLD_1'])
                        plt.plot(train_val_df.groupby(['dept_id']).sum().loc['HOUSEHOLD_2'])
                        plt.legend(['HOUSEHOLD_1','HOUSEHOLD_2'])
                        axes.xaxis.set_major_locator(MaxNLocator(15)) 
                        axes.yaxis.set_major_locator(MaxNLocator(15))
                        axes.set_title("Number of sales by department household per day",fontsize= 18)
                        axes.set_xlabel("Days", fontsize= 18)
                        axes.set_ylabel("Number of department household", fontsize= 18)
                        plt.show()
                        st.pyplot()


        elif feat == "dept_id":  
                train_val_df = decompress_pickle('train_val_df.pbz2')
                if st.button("Generate plot for department"):
                    temp4 = train_val_df.groupby(['dept_id'])['item_id'].count()
                    plt.figure(figsize=(10,6))
                    temp4.plot(kind='bar')
                    plt.title("Number of sales by Department")
                    st.pyplot() 

        else:
                train_val_df = decompress_pickle('train_val_df.pbz2')
                if st.button("Generate plot for store"):
                    temp2 = train_val_df.groupby(['store_id'])['state_id'].count()
                    plt.figure(figsize=(10,6))
                    temp2.plot(kind='bar',color=['r', 'g', 'b'])
                    plt.title("Number of sales by State")
                    st.pyplot() 

            

    if choice == 'Data Visualization':   
        train_val_df = decompress_pickle('train_val_df.pbz2')
        category = train_val_df.groupby(['cat_id']).sum()
        
        if st.button("Generate plot for Distribution of State, Category, Department"): 
            
            ### Sales Distribution across Category and Departments

            sales_agg_cat = train_val_df.groupby(["state_id","cat_id","dept_id"])[category.columns].sum()
            sales_agg_cat = pd.DataFrame(sales_agg_cat.sum(axis=1)).reset_index().rename({0:"sales"}, axis=1)

            # Subplots and Pie charts with plotly

            fig = make_subplots(rows=1, cols=3,specs=[[{'type':'domain'},{'type':'domain'}, {'type':'domain'}]])
            fig.add_trace(go.Pie(values=sales_agg_cat['sales'], labels=sales_agg_cat['state_id']), 1,1)
            fig.add_trace(go.Pie(values=sales_agg_cat['sales'], labels=sales_agg_cat['cat_id']), 1,2)
            fig.add_trace(go.Pie(values=sales_agg_cat['sales'], labels=sales_agg_cat['dept_id']), 1,3)

            # updating traces and layout

            fig.update_traces(hole=.4, hoverinfo="label+percent+name")

            fig.update_layout(
                title_text="Percentage Sales Distribution across States, Category and Departments",
                annotations=[dict(text='State', x=0.12, y=0.5, font_size=12, showarrow=False),
                    dict(text='Category', x=0.5, y=0.5, font_size=12, showarrow=False),
                             dict(text='Department', x=0.91, y=0.5, font_size=12, showarrow=False)])
            #fig.show()
            st.plotly_chart(fig)


        if st.button("Generate plot for Distribution by year"):  
            
            sales_train1 = pd.DataFrame(train_val_df[category.columns].T.sum(axis=1)).rename({0:"sales"},axis=1).merge(calendar.set_index("d"),how="left", left_index=True,right_index=True, validate="1:1")

            sales_train_year = sales_train1.groupby(["year"])["sales"].sum()
            sales_train_year = pd.DataFrame(sales_train_year)

            fig = px.bar(sales_train_year.reset_index(), x="year", y="sales", color="year", text="sales", title="Year-wise walmart sales Contribution")
            fig.update_traces(texttemplate='%{text:.2s}', textposition='auto')
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
            #fig.show()
            st.plotly_chart(fig)

        if st.button("Generate plot for Distribution by month"): 
            sales_train1 = pd.DataFrame(train_val_df[category.columns].T.sum(axis=1)).rename({0:"sales"}, axis=1).merge(calendar.set_index("d"),how="left", left_index=True,
                                             right_index=True, validate="1:1")
            sales_train_month = sales_train1.groupby(["month"])["sales"].sum()
            sales_train_month = pd.DataFrame(sales_train_month)

            fig = px.bar(sales_train_month.reset_index(), x="month", y="sales", color="month", text="sales", title="Month-wise Walmart sales contribution")
            fig.update_traces(texttemplate='%{text:.2s}', textposition='auto')
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
            st.plotly_chart(fig)


        if st.button("Generate plot for Distribution by weekday"):  
            sales_train1 = pd.DataFrame(train_val_df[category.columns].T.sum(axis=1)).rename({0:"sales"}, axis=1).merge(calendar.set_index("d"),how="left", left_index=True,
                                             right_index=True, validate="1:1")
            sales_train_weekday= sales_train1.groupby(["month", "weekday"])["sales"].sum()
            sales_train_weekday = pd.DataFrame(sales_train_weekday)

            fig = px.bar(sales_train_weekday.reset_index(), x="month", y="sales", color="weekday", text="sales", title="Monthly Sales bifurcated by weekdays")
            fig.update_traces(texttemplate='%{text:.2s}', textposition='auto')
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
            st.plotly_chart(fig)



    if choice == 'Modeling':
        output = decompress_pickle('output.pbz2') 
        st.write("Data read successfully")
        st.write(output.shape)
        item = st.selectbox('Select the item you want to predict sales for:',('FOODS_2_283_CA_2_validation', 'FOODS_2_070_TX_3_validation', 'FOODS_2_026_WI_1_validation', 'HOBBIES_2_038_CA_3_validation', 'FOODS_3_394_TX_1_validation', 'HOUSEHOLD_1_139_CA_2_validation', 'HOUSEHOLD_2_458_TX_2_validation', 'HOUSEHOLD_1_184_TX_2_validation', 'HOUSEHOLD_1_134_TX_3_validation', 'FOODS_3_678_CA_3_validation', 'HOUSEHOLD_1_046_WI_3_validation', 'FOODS_3_689_CA_4_validation', 'FOODS_3_605_CA_4_validation', 'HOUSEHOLD_1_336_TX_2_validation', 'FOODS_2_001_WI_2_validation', 'HOBBIES_1_218_CA_4_validation', 'FOODS_3_793_CA_1_validation', 'FOODS_2_318_CA_4_validation', 'HOUSEHOLD_1_390_WI_2_validation', 'HOUSEHOLD_1_447_CA_1_validation', 'FOODS_3_485_CA_4_validation', 'FOODS_3_688_WI_2_validation', 'HOBBIES_1_155_WI_1_validation', 'FOODS_2_047_WI_3_validation', 'FOODS_3_351_TX_3_validation', 'FOODS_3_761_TX_2_validation', 'FOODS_2_052_CA_4_validation', 'HOBBIES_2_029_TX_3_validation', 'HOUSEHOLD_2_176_TX_3_validation', 'FOODS_3_548_WI_2_validation'))
        
        forecast = output.loc[output['item id'] == item]
        st.dataframe(forecast)
        plot_forecast(forecast.iloc[0, 2:], item)
        
    if choice == 'About':
        st.write('Author:')
        col1, mid, col2 = st.beta_columns([1,10,20])
        with col1:
            st.image('profile.png', width=200)
        with col2:
            st.write('My Name is Sonal Jain. I am a Data Science graduate from Northeastern University, Boston. I spend my time wrangling data, discovering patterns, analyzing datasets, and build models to make real-world decisions. I can make your data tell a story.')
            

if __name__ == "__main__":
    main()

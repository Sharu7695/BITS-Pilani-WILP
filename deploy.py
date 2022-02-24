



import streamlit as st
from streamlit import caching
import pandas as pd
import numpy as np
import datetime
from datetime import datetime
from datetime import date
import pickle
from sklearn.linear_model import SGDRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from PIL import Image

from optimal_time_deploy_func import data_cleaner
from optimal_time_deploy_func import time_pred

from price_deploy_func import price_cleaner
from price_deploy_func import price_pred

# def cache_clear_dt(dummy): # 👈 Changed this
#    clear_dt = date.today()
#    return clear_dt
# if cache_clear_dt("dummy")<date.today():
#    caching.clear_cache()

data_o = pd.read_csv("origin_city.csv") 
gg = sorted(list(set(list(data_o.origin_city))))

data_d = pd.read_csv("dest_city.csv")  
hh = sorted(list(set(list(data_d.dest_city))))
# 

X_train = pd.read_csv("X_train.csv") 
X_train.drop(['Unnamed: 0'],axis=1,inplace=True)

price_X_train = pd.read_csv("price_X_train.csv") 
price_X_train.drop(['Unnamed: 0'],axis=1,inplace=True)

pickle_time = open("optimal_time_sgd.pkl","rb")
# pickle_time = open("new_dtree_model.pkl","rb")
model_opt_time = pickle.load(pickle_time)

pickle_price = open("price_rf.pkl","rb")
model_price = pickle.load(pickle_price)

#@st.cache # 👈 Changed this
#@st.cache(suppress_st_warning=True)  
def main():

    st.sidebar.title("Let's Travel")
    st.sidebar.markdown("Please enter your journey details:")
    o_city = st.sidebar.selectbox('Origin City', gg)
    d_city = st.sidebar.selectbox('Destination City',hh)
    
    #date = st.sidebar.date_input('Inbound Date')
    date1 = st.sidebar.date_input('Inbound Date')
    #input = datetime.strptime(str(date1), '%Y-%m-%d').strftime('%d/%m/%Y') 
    today = date.today()
    #str(datetime.strptime(str(date.today()), '%Y-%m-%d').strftime('%d/%m/%Y'))
    #date.today()
    #str(datetime.datetime.strptime(str(date.today()), '%Y-%m-%d').strftime('%d/%m/%Y'))
    
    #str(date.today().strftime('%Y-%m-%d'))
    date_f = ""
    if date1 < today:
        st.error("Enter the future date")
        #date = st.sidebar.date_input('Inbound Date',)
    #else:
    date_f = date1
    f_type = st.sidebar.radio('Flight type', ["cheap","best","fast"])
    #st.sidebar.select_slider('Slide to select the flight type', options=["cheap","best","fast"])
    stops= st.sidebar.radio('No. of Stops', ["direct","1 stop","2 stops"])
    airline = st.sidebar.selectbox('Preferred Airline',["IndiGo","GoAir","Air India","SpiceJet","AirAsia India","Vistara","Multiple Airlines"])
    #st.sidebar.selectbox("Flight Type",["Cheap","Best","Fast"])
    
    
    #### Input from users to pass in the test set
    Origin = list(data_o.loc[data_o['origin_city'] == o_city, "origin_code"])[0]
    Destination =  list(data_d.loc[data_d['dest_city'] == d_city, "dest_code"])[0]
    Date = datetime.strptime(str(date_f), '%Y-%m-%d').strftime('%d/%m/%Y') 
    Airline = airline
    sort = f_type
    Stops = stops
    
        ## PREDICTING OPTIMAL TIME
    #1
    test_df = data_cleaner(Date,Origin,Destination,Airline,sort,Stops)
    
    #2
    test_df_pca = time_pred(X_train,test_df)
    
    #3
    
    test_df_pred = model_opt_time.predict(test_df_pca)
    Optimal_time = test_df_pred
    
#       ## PREDICTING PRICE FOR THE OPTIMAL TIME
#   #1
    price_test_df = price_cleaner(Date,Origin,Destination,Airline,sort,Stops)
    
    #2
    price_test_df_pca = price_pred(price_X_train,price_test_df)
    
    #3
    price_test_df_pred = model_price.predict(price_test_df_pca)
    
    x, y, z = st.beta_columns([1,3,2])
    with x:
        image = Image.open('gg.png')
        st.image(image, caption='Travel Safe')
#         st.image('new_gg.png')
    with y:
        st.title("Your Travel Partner")
    
    opt_time = ""
    app_price = ""
    if st.sidebar.button('Predict'):
#         st.write(Date)
#         st.write(Origin)
#         st.write(Destination)
#         st.write(Airline)
#         st.write(sort)
#         st.write(Stops)
#         st.write(test_df_pca.shape)
#        
        opt_time = int(abs(Optimal_time))
	#opt_time = abs(opt_time)
#         opt_time = 2500
#         app_price = 15432
        app_price = int(abs(price_test_df_pred/100))
	#app_price = int(app_price)
    
        
        st.success('The Optimal Time for the flight is {} hrs i.e {} days'.format(round(opt_time/(60*60)),round(opt_time/(24*60*60))))

        st.success("The price for the chosen flight is approximately Rs {}".format(app_price))
        
        
        b,c = st.beta_columns([1,2])
        with c:
            image1 = Image.open('img2.png')
            st.image(image1,width = 300)
#             st.image('img2.png', width = 300)
    else:
        with y:
            image2 = Image.open('img1.jpg')
            st.image(image2,width = 400)
#             st.image('img1.jpg', width =400)
        
if __name__ == '__main__':
    main()





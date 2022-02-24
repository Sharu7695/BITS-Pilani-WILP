

# In[1]:


import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from datetime import datetime


# In[3]:

def price_cleaner(Date,Origin,Destination,Airline,sort,Stops):
    #Creating empty DataFrame for holding models information
    cols = ['Date','Origin','Destination','Airline','sort','Stops']
    test = pd.DataFrame(columns = cols)
    test.loc[0] = [Date, Origin, Destination, Airline, sort, Stops]
    test = test.rename(columns = {'Stops' : 'Out_Journey_Type'})
    
    new_DATE = datetime.strptime(Date,'%d/%m/%Y').date()
    Weekday = new_DATE.strftime("%A")
    Day = int(new_DATE.strftime("%d"))
    Month = int(new_DATE.strftime("%m"))
    
    test['Out_Day'] = Day
    test['Out_Month'] = Month
    test['Out_Weekday'] = Weekday
    test = test.drop('Date',1) #We no longer need the date column
    
    ## Let us create dummies for categorical columns
    dum = test[["Airline","Origin","Destination"]]
    dummy = pd.get_dummies(dum, prefix=["Out_Airline",'Out_Cities','Dest_Cities'])

#     dummy = pd.get_dummies(dum, prefix=["Airline",'origin','Dest'])
    
    price_test_df = pd.concat([test,dummy],1) # Let us concatenate dummy columns with original dataset

    price_test_df.drop(dum.columns,1,inplace=True) #Let us drop the categorical columns whose dummies are ceated
    
        
    # Mapping the categorical columns by numerical values
    price_test_df['Out_Journey_Type'].replace({"direct": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3}, inplace = True)
    price_test_df['sort'].replace({"cheap": 0, "best": 1, "fast": 2}, inplace = True)
    price_test_df['Out_Weekday'].replace({"Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4, "Friday": 5, "Saturday": 6, "Sunday": 7}, inplace = True)

    return price_test_df


def price_pred(price_X_train,price_test_df):
    
    # Let us 1st take care of the extra columns in our original X_train
    extra_col =[]
    for i in price_X_train.columns:
        if i not in price_test_df.columns:
            extra_col.append(i)
            
    # Let us insert these missing columns in test_df & impute them with 0
    for i in extra_col:
        price_test_df[i] = np.nan
    price_test_df = price_test_df.fillna(0)
    
    ## Feature scaling
    # let us select the numerical columns that need scaling
    num_col = ['Out_Day', 'Out_Weekday', 'Out_Month', 'Out_Travel_Time',
               'Out_Journey_Type', 'sort', 'Out_hour', 'Out_min']

    ## Scaling the train and test data
    sc = StandardScaler()
    price_X_train[num_col] =  sc.fit_transform(price_X_train[num_col])
    # Let us transform scaling on this test set
    price_test_df[num_col] = sc.transform(price_test_df[num_col])
    
    # Let us apply PCA on training & test data
    pca_final = IncrementalPCA(n_components = 37)
    price_X_train_pca = pca_final.fit_transform(price_X_train)
    price_test_df_pca = pca_final.transform(price_test_df)  ## on user input dataset

    return price_test_df_pca



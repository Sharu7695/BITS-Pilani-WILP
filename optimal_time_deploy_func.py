
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import IncrementalPCA
from sklearn.model_selection import train_test_split
from datetime import datetime


def X_train_maker(df):
    
    # Let us drop the unnamed column
    df.drop(['Unnamed: 0'],axis=1,inplace=True)
    # Let us correct the datatypes for features that have been wrongly identified
    df = df.astype({"Out_Day":'int64', "out_Year":'int64','Return_Day':'int64','Return_Year':'int64','Price':'int64'})
    df['Out_Month'] = pd.to_datetime(df.Out_Date,format="%d/%m/%Y").dt.month #Converting the Out_Month to the month number.
    df["Price"] = df["Price"]*0.56 #Reducing the price to one way price
    df.drop(['Return_Date'], axis=1, inplace=True) #Dropping the return type.
    df['timestamp'] = pd.to_datetime(df['timestamp'], infer_datetime_format=True) #Converting the timestamp to date and time 

    # Get the departure time and skiping the arrival time as it can be predict from the Out_Travel_Time
    time = list(df["Out_Time"])
    duration=[]
    for i in range(len(time)):
        duration.append(time[i].split(sep='–')[0])
    df["Out_Time"]=duration

    # Adding sec to Out_Time and storing it in the Out_Time1 for later use 
    time = list(df['Out_Time']) 
    for i in range(len(time)):
        time[i] = time[i] + ':00'
    df['Out_Time1'] = time

    # Adding 0 to the Out_Day from 1-9 to convert it to timestamp to get the target_Var
    d1 = list(df['Out_Day'])
    for i in range(len(d1)):
        if d1[i]>=1 and d1[i]<=9:
            d1[i] = '0'+str(d1[i])
    df['Out_Day1'] = d1

    # Adding 0 to the Out_Month from 1-9 to convert it to timestamp to get the target_Var
    d1 = list(df['Out_Month'])
    for i in range(len(d1)):
        if d1[i]>=1 and d1[i]<=9:
            d1[i] = '0'+str(d1[i])
    df['Out_Month1'] = d1

    # Getting the timestamp in object format.
    t1 = list(df['Out_Day1'])
    t2 = list(df['Out_Month1'])
    t3 = list(df['Out_Time'])
    for i in range(len(t1)):
        t1[i] = '2021'+str(t2[i])+str(t1[i])+'-'+t3[i].split(':')[0]+t3[i].split(':')[1]
    df['Out_Timestamp'] = t1

    # Converting to timestamp
    df['Out_Timestamp'] = pd.to_datetime(df['Out_Timestamp'], infer_datetime_format=True)

    #Finding the difference between the Timestamp and Out_Timestamp to get the Target_Var
    timestamp1 = list(df['timestamp'])
    timestamp2 = list(df['Out_Timestamp'])
    diff = []
    for i in range(len(timestamp1)):
        diff.append(timestamp2[i]-timestamp1[i])
        
    df['Optimal_time'] = diff 
    
    # # Droping the non required columns.
    del_cols = ['Out_Time1','Out_Timestamp','Out_Date','Out_Month1','Out_Day1']
    df.drop(del_cols,axis=1,inplace=True) 

    # Extracting hours and minutes and separating it in the separate columns.
    # Extracting Hours
    df["Out_hour"] = pd.to_datetime(df["Out_Time"]).dt.hour

    # Extracting Minutes
    df["Out_min"] = pd.to_datetime(df["Out_Time"]).dt.minute

    df.drop(["Out_Time","Return_Time"],axis=1,inplace=True)
    
    df.drop(['Out_Stop_Cities','Return_Stop_Cities','out_Year','Return_Year'],axis=1,inplace=True)

    ## Time taken by plane to reach destination is called Duration
    # It is the differnce betwwen Departure Time and Arrival time
    duration = list(df["Out_Travel_Time"])

    duration_hours = []
    duration_mins = []
    for i in range(len(duration)):
        duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
        duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))

    # Converting the duration in minutes and storing it in the Out_Travel_Time

    for i in range(len(duration_hours)):
        duration_hours[i] = duration_hours[i]*60
        duration_mins[i] = duration_mins[i] + duration_hours[i]
    df['Out_Travel_Time'] = duration_mins 

    col = ["Return_Travel_Time","Return_Month"]
    df.drop(col, axis = 1, inplace = True)
    
    df = df.astype({'Optimal_time':'object'})

    # Converting the Target_Var to minutes.

    t = list(df['Optimal_time'])
    t4 = []
    for i in range(len(t)):
        t1 = str(t[i])
        t2 = t1.split()
        t3 = t2[2].split(':')
        t4.append((int(t2[0])*1440)+(int(t3[0])*60)+int(t3[1]))
    df['Optimal_time'] = t4
       
    import fuzzywuzzy
    from fuzzywuzzy import process
    sort = df['sort'].unique()

    matches = fuzzywuzzy.process.extract("cheap", sort, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)
    def replace_matches_in_column(df, column, string_to_match, min_ratio = 100):
        # get a list of unique strings
        strings = df[column].unique()

        # get the top 10 closest matches to our input string
        matches = fuzzywuzzy.process.extract(string_to_match, strings, 
                                             limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

        # only get matches with a ratio > 90
        close_matches = [matches[0] for matches in matches if matches[1] >= min_ratio]

        # get the rows of all the close matches in our dataframe
        rows_with_matches = df[column].isin(close_matches)

        # replace all rows with close matches with the input matches 
        df.loc[rows_with_matches, column] = string_to_match

        # let us know the function's done
        print("All done!")

    replace_matches_in_column(df=df, column='sort', string_to_match="cheap")

    matches = fuzzywuzzy.process.extract("best", sort, limit=1, scorer=fuzzywuzzy.fuzz.token_sort_ratio)
    replace_matches_in_column(df=df, column='sort', string_to_match="best")
    
    
    ## Consider only the code of the place and removing the rest like the name or the +1 in the beginning.
    c = list(df["Out_Cities"])
    cities = []
    for i in range(len(c)):
        if len(c[i].split()) == 2:
            if ('+1' in c[i]) or ('+2' in c[i]):
                cities.append(c[i].split()[1])
            else:
                cities.append(c[i].split()[0])
    df["Out_Cities"] = cities
    
    # Doing the same for the return cities.
    c = list(df["Return_Cities"])
    cities = []
    for i in range(len(c)):
        if len(c[i].split()) == 2:
            if ('+1' in c[i]) or ('+2' in c[i]):
                cities.append(c[i].split()[1])
            else: 
                cities.append(c[i].split()[0])
    df["Return_Cities"] = cities
    
    ## As the returning cities are our destination so are changing the name to 'Dest_Cities'.
    df.rename(columns = {'Return_Cities':'Dest_Cities'},inplace=True)
    
    dum = df[["Out_Airline","Out_Cities","Dest_Cities"]]
#     dummy = pd.get_dummies(dum, prefix=["Airline",'origin','Dest'],drop_first = False)
    dummy = pd.get_dummies(dum, drop_first = False)
    
    data = pd.concat([df, dummy], axis=1)
    data.drop(dum.columns,1,inplace=True)
    
    # Mapping the categorical columns by numerical values
    data['Out_Journey_Type'].replace({"direct": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3}, inplace = True)
    data['sort'].replace({"cheap": 0, "best": 1, "fast": 2}, inplace = True)
    data['Out_Weekday'].replace({"Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4, "Friday": 5, "Saturday": 6, "Sunday": 7}, inplace = True)

    drop_col = ['Return_Airline','Return_Day','Return_Weekday','Return_Journey_Type','timestamp']
    data.drop(drop_col, axis=1, inplace=True)
    
#     data["AvgPrice"] = (data.groupby("sort")["Price"].transform("mean"))
    
    # Putting feature variables to X
    X = data.copy()

    # Putting response variable to y
    y = X.pop('Optimal_time')

    # Let us split our dataset into train & yest
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7 , random_state=100)
    
    return X_train,y_train,X_test,y_test



# In[6]:


def data_cleaner(Date,Origin,Destination,Airline,sort,Stops):
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
    
    test_df = pd.concat([test,dummy],1) # Let us concatenate dummy columns with original dataset

    test_df.drop(dum.columns,1,inplace=True) #Let us drop the categorical columns whose dummies are ceated
    
        
    # Mapping the categorical columns by numerical values
    test_df['Out_Journey_Type'].replace({"direct": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3}, inplace = True)
    test_df['sort'].replace({"cheap": 0, "best": 1, "fast": 2}, inplace = True)
    test_df['Out_Weekday'].replace({"Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4, "Friday": 5, "Saturday": 6, "Sunday": 7}, inplace = True)

    return test_df


# In[7]:



def time_pred(X_train,test_df):
    
    # Let us 1st take care of the extra columns in our original X_train
    extra_col =[]
    for i in X_train.columns:
        if i not in test_df.columns:
            extra_col.append(i)
            
    # Let us insert these missing columns in test_df & impute them with 0
    for i in extra_col:
        test_df[i] = np.nan
    test_df = test_df.fillna(0)
    
    ## Feature scaling
    # let us select the numerical columns that need scaling
    num_col = ['Out_Day', 'Out_Weekday', 'Out_Month', 'Out_Travel_Time',
               'Out_Journey_Type', 'sort', 'Out_hour', 'Out_min','Price']

    ## Scaling the train and test data
    sc = StandardScaler()
    X_train[num_col] =  sc.fit_transform(X_train[num_col])
    # Let us transform scaling on this test set
    test_df[num_col] = sc.transform(test_df[num_col])
    
    # Let us apply PCA on training & test data
    pca_final = IncrementalPCA(n_components = 37)
    X_train_pca = pca_final.fit_transform(X_train)
    test_df_pca = pca_final.transform(test_df)  ## on user input dataset

    return test_df_pca

# def time_pred(X_train,test_df):
#         # Let us 1st take care of the extra columns in our original X_train
#     extra_col =[]
#     for i in X_train.columns:
#         if i not in test_df.columns:
#             extra_col.append(i)        
#     #Let us insert these missing columns in test_df & impute them with 0
#     for i in extra_col:
#         test_df[i] = np.nan
#     test_df = test_df.fillna(0)
#     y_train = pd.read_csv("y_train.csv") 
#     y_train.drop(['Unnamed: 0'],axis=1,inplace=True)
#     fs = SelectKBest(score_func=mutual_info_regression, k=10)
#     # learn relationship from training data
#     fs.fit(X_train,y_train)
#     # transform train input data
#     X_train_fs = fs.transform(X_train)
#     test_df_fs = fs.transform(test_df)
#     return test_df_fs
    







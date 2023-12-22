### Import libriries

from datetime import datetime, timedelta,date
import pandas as pd
#%matplotlib inline
#import matplotlib.pyplot as plt
import numpy as np
#from __future__ import division
import warnings
warnings.filterwarnings("ignore")

#import plotly.plotly as py
#from chart_studio.plotly import plot, iplot as py
#import plotly.offline as pyoff
#import plotly.graph_objs as go
#pyoff.init_notebook_mode()

import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping
from keras.utils import np_utils
from tensorflow.keras.layers import LSTM
from sklearn.model_selection import KFold, cross_val_score, train_test_split

import pandas as pd
import os



#####################################################################################

#here we decide if we train again the model or we call the saved model

train = True # Choose True/False

#####################################################################################


#Functions to calculate the features 'int_weekday' and 'int_week'

def calculate_weekday(cur_weekday):
  #Assign an integer according to the day of the week (e.g., Monday --> 0, Tuesday --> 1)
  int_cur_weekday = 0 if cur_weekday == 'Δευ' else \
                    (1 if cur_weekday == 'Τρί' else \
                      (2 if cur_weekday == 'Τετ' else \
                        (3 if cur_weekday == 'Πέμ' else \
                          (4 if cur_weekday == 'Παρ' else 5))))
  return int_cur_weekday

def calculate_week(cur_date):
  #Assign an integer according to the week of the month (e.g., 1-7 --> 0, 8-14 --> 1)
  temp_current_date = cur_date.ctime().split(' ')[2]
  if temp_current_date != '':
    curr_date = int(temp_current_date)
  else:
    curr_date = int(cur_date.ctime().split(' ')[3])

  int_cur_week = 0 if 1 <= curr_date <= 7 else \
                    (1 if 8 <= curr_date <= 14 else \
                      (2 if 15 <= curr_date <= 21 else \
                        (3 if 22 <= curr_date <= 28 else 4)))  
  return int_cur_week




#####################################################################################


### read data and rename the basic columns 
calendar= pd.read_excel(r'C:\Python Projects\Everyday_sales_prediction\Historical Data\calendar.xlsx')



#path = os.getcwd()
path = 'C:\Python Projects\Everyday_sales_prediction\Historical Data'
files = os.listdir(path)
print(files)

files_xls = [f for f in files if f[0:5] == 'Sales']

d = pd.DataFrame()

for f in files_xls:
        dat = pd.read_excel(r'C:\Python Projects\Everyday_sales_prediction\Historical Data\{}'.format(f), 'Sheet1')
        d = d.append(dat)


data = d

print(data)

#data= pd.read_excel('Sales_2018_1-2022_6.xlsx')


data = data.rename(columns={"Ημερομηνία": "date", "Πωλήσεις": "sales"})

### Join the main dataset with a calender and fill NaN with 0
df = pd.merge(calendar,data,on='date',how='left')     #https://www.analyticsvidhya.com/blog/2020/02/joins-in-pandas-master-the-different-types-of-joins-in-python/

### !OPTINAL Choose a specific weekday to exclude
#day = 'Δευ'
#day = 'Τρι'
#day = 'Τετ'
#day = 'Πεμ'
#day = 'Παρ'
#day = 'Σαβ'
excluded_day = 'Κυρ'

df = df.loc[(df["weekday"] != excluded_day)].reset_index()

df['sales'] = df['sales'].fillna(0) #https://www.geeksforgeeks.org/replace-nan-values-with-zeros-in-pandas-dataframe/

df = df.drop(['index', 'Εβδομάδα', 'Ετος', 'Μήνας', 'Ημέρα Εβδομάδος'], axis=1)

#Create new columns 'int_weekday', 'int_week'
df['int_weekday'] = np.nan
df['int_week'] = np.nan

for row in range(len(df)):

  #Replace 0 values or outliers (<100000), with the mean of the corresponding day of the previous and next week.
  current_sales = df.loc[row]['sales'].copy()
  if current_sales == 0 or current_sales < 100000:
    if row-6 >= 0 and row+6 <= len(df)-1:
      if df.loc[row-6]['sales'] >= 100000 and df.loc[row+6]['sales'] >= 100000:
        df['sales'][row] = (df.loc[row-6]['sales']+df.loc[row+6]['sales'])/2  
      elif df.loc[row-6]['sales'] < 100000 and df.loc[row+12]['sales'] >= 100000:
        df['sales'][row] = (df.loc[row+6]['sales']+df.loc[row+12]['sales'])/2
      elif df.loc[row+6]['sales'] < 100000 and df.loc[row-12]['sales'] >= 100000:
        df['sales'][row] = (df.loc[row-6]['sales']+df.loc[row-12]['sales'])/2
      else:
        print("There are cases that have not been taken into account!!! Case 1!!!")  
        break
    elif row-6 < 0 and df.loc[row+6]['sales'] >= 100000 and df.loc[row+12]['sales'] >= 100000:
      df['sales'][row] = (df.loc[row+6]['sales']+df.loc[row+12]['sales'])/2
    elif row+6 > len(df)-1 and df.loc[row-6]['sales'] >= 100000 and df.loc[row-12]['sales'] >= 100000:
      df['sales'][row] = (df.loc[row-6]['sales']+df.loc[row-12]['sales'])/2
    else:
      print("There are cases that have not been taken into account!!!")
      break
  
  current_weekday = df.loc[row]['weekday']
  df['int_weekday'][row] = calculate_weekday(current_weekday)

  if np.isnan(df['int_weekday'][row]):
    print("There are cases that have not been taken into account!!! Case 2!!!") 

  df['int_week'][row] = calculate_week(df.loc[row]['date'])

  if np.isnan(df['int_week'][row]):
    print("There are cases that have not been taken into account!!! Case 3!!!")
    


#####################################################################################


#df = df.loc[df.date <= '2022-05-08']

#df = df.loc[df.date < '2023-06-24']

today = date.today()

df = df.loc[df.date < '{}'.format(today)]

print(df)


#####################################################################################

"""
unknown_actual_sales = df['sales'].iloc[-1:].tolist()
#dates = df['date'].iloc[-1:]
dates = df.iloc[-1:, [0,1]]
df = df.iloc[:-1]
df.tail(10)
"""

#####################################################################################


######Choose the features (except lags)#############!!!!!!!!!
features = ['int_weekday', 'int_week']
df_sales = pd.DataFrame(df, columns= ['date','sales']+[feat for feat in features])




#####################################################################################


#create a new dataframe to model the difference
df_diff = df_sales.copy()
#add previous sales to the next row
df_diff['prev_sales'] = df_diff['sales'].shift(1)
#drop the null values and calculate the difference
df_diff = df_diff.dropna()
df_diff['diff'] = (df_diff['sales'] - df_diff['prev_sales'])


#####################################################################################


df_supervised = df_diff.drop(['prev_sales'],axis=1)

###Select the window length#####!!!!!!!!!!!!
#window_length = 6

window_length = 24

print('window length is: ', window_length)

#adding lags
for inc in range(1, window_length): #!!!how large is the window
    field_name = 'lag_' + str(inc)
    df_supervised[field_name] = df_supervised['diff'].shift(inc)
    for feat in features:
      df_supervised[field_name+'_'+feat] = df_supervised[feat].shift(inc)  
#drop null values
df_supervised = df_supervised.dropna().reset_index(drop=True)
#Drop the selected features reffered to 'diff'
df_supervised = df_supervised.drop([feat for feat in features],axis=1)




#####################################################################################


#import MinMaxScaler and create a new dataframe for LSTM model
from sklearn.preprocessing import MinMaxScaler
df_model = df_supervised.drop(['sales','date'],axis=1)

#split train and test set. It is important the first feature to be the 'diff' !!!!
#Choose the day from which the test set should be began!
day_begin_test = -6
train_set, test_set = df_model[0:day_begin_test].values, df_model[day_begin_test:].values

#apply Min Max Scaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(train_set)
# reshape training set
train_set = train_set.reshape(train_set.shape[0], train_set.shape[1])
train_set_scaled = scaler.transform(train_set)
# reshape test set
test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
test_set_scaled = scaler.transform(test_set)




#####################################################################################



#Reshape train and test set to match the shape [batch, timestep, data_dim]. 
#The first column ('diff') will be used as the ground truth.
#Each 'lag' should be considered as a timestep and it consists of (the 'data_dim' dimension) 
#the corresponding value of the difference in sales as well as the selected features.
X_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1]
X_train = X_train.reshape(X_train.shape[0], int(X_train.shape[1]/(len(features)+1)), len(features)+1)
X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1]
X_test = X_test.reshape(X_test.shape[0], int(X_test.shape[1]/(len(features)+1)), len(features)+1)


  
model = Sequential()
model.add(LSTM(20, stateful=False)) #batch_input_shape=(batch_size, X_train.shape[1], X_train.shape[2])
model.add(Dense(1))

if train:
  
  #Select batch_size
  batch_size=1

  #Save model at
  checkpoint_path = "./training_1/cp.ckpt"
  checkpoint_dir = os.path.dirname(checkpoint_path)

  cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                   save_weights_only=True,
                                                   verbose=1)
  
  opt = Adam(learning_rate=0.001)
  model.compile(loss='mean_squared_error', optimizer=opt)

  #model.fit(X_train, y_train, epochs=1, batch_size=batch_size, verbose=1, shuffle=True, validation_data = (X_test, y_test))

  model.fit(X_train, y_train, epochs=200, batch_size=batch_size, verbose=1, shuffle=True, validation_data = (X_test, y_test), callbacks=[cp_callback]) 


#####################################################################################

  loss_history = model.history.history['loss']
  val_loss_history = model.history.history['val_loss']
  epochs = np.arange(1, len(loss_history)+1)


#####################################################################################

else:
  print("A pre-trained model is loaded!")
  checkpoint_path = "./training_baseline/cp.ckpt"
  model.load_weights(checkpoint_path)
  print("Loading successful!")

#####################################################################################

####Make predictions for test set####
y_pred = model.predict(X_test,batch_size=1)

#fill y_pred with dummy zeros for the 5 remaining columns
y_pred_filled = np.append(y_pred, np.zeros((y_pred.shape[0],X_test.shape[1]*X_test.shape[2])), axis=1)

#Invert scaling
pred_test_set_inverted = scaler.inverse_transform(y_pred_filled)

#use the correct dates to plot for predicted values
next_dates_list = list(df_sales[day_begin_test:].date)

#create dataframe that shows the predicted sales
result_list = []
sales_dates = next_dates_list
act_sales = list(df_sales[day_begin_test:].sales)
for index in range(pred_test_set_inverted.shape[0]):
    result_dict = {}
    result_dict['pred_value'] = int(pred_test_set_inverted[index][0] + act_sales[index])
    result_dict['date'] = sales_dates[index]
    result_list.append(result_dict)
    
df_result = pd.DataFrame(result_list)




#####################################################################################


###Prediction for unknown days starts here#####
from collections import deque

#Get only the last window from the test set (X_test) to perform predictions for the unkown days. Note that we exclude the last element from that window
#and we add at the first position the last element of the y_test which corresponds to the difference of the last day of the test set with the previous day.  
#We assume that the last element of y_test corresponds to the column 'diff' of the last row of 'df_diff'. Therefore, we get this along with its features 
#(which corresponds to the same day as 'diff') from 'df_diff' using df_diff.iloc[-1]['diff'] and [df_diff.iloc[-1][feat] for feat in features].
#Also, we need to scale these values using the 'scaler'.
last_elem_y_test_features = [0, df_diff.iloc[-1]['diff']]+[df_diff.iloc[-1][feat] for feat in features]+([0]*(train_set_scaled.shape[1]-(2+len(features))))
scaled_last_elem_y_test_features = scaler.transform(np.array([last_elem_y_test_features]))[:, 1:2+len(features)]
X_test_new = np.append(scaled_last_elem_y_test_features, X_test[-1, 0:-1].copy(), axis=0)

#Now we need to calculate the dates of the days for which we are going to perform the predictions, in order to get their corresponding features.
#This is neccesary for the features 'int_weekday' and 'int_week'. Also we need these for the plot.
import datetime
sales_dates = list(df_sales[-1:].date)
#Check if the next day is equal to the excluded day. If so, we should store the date of the second following day. 
#Note that instead of checking the date of the next day (which is not included in the 'calendar') 
#we check the date of the 6th day before (which should have the same day name, that is day_name+1=day_name-6)
date_shift = 1 if calendar[calendar['date']==(sales_dates[0]+datetime.timedelta(days=-6)).to_datetime64()]['weekday'].values[0] != excluded_day else 2
next_day_date = sales_dates[0]+datetime.timedelta(days=date_shift)
next_dates_list = [next_day_date]
#As the date of the next day is already calculated, now we should use this day along with the 'timedelta(days=-7)'.
next_days_names = [calendar[calendar['date']==next_day_date+datetime.timedelta(days=-7)]['weekday'].values[0]]
for day_date in range(1,pred_test_set_inverted.shape[0]):
  date_shift = 1 if calendar[calendar['date']==(next_day_date+datetime.timedelta(days=-6)).to_datetime64()]['weekday'].values[0] != excluded_day else 2
  next_dates_list.append(next_day_date+datetime.timedelta(days=date_shift))
  next_day_date = next_dates_list[day_date]
  next_days_names.append(calendar[calendar['date']==next_day_date+datetime.timedelta(days=-7)]['weekday'].values[0])

#Calculate the features for the next days and scale them
next_days_features = dict(zip(features, [[] for _ in features]))
scaled_next_days_features = dict(zip(features, [[] for _ in features]))
for day in range(len(next_days_names)):
  for feat in features:
    if 'int_weekday'==feat:
      next_days_features[feat].append(calculate_weekday(next_days_names[day]))
    if 'int_week'==feat:
      next_days_features[feat].append(calculate_week(next_dates_list[day]))
  temp_unscaled_features = [0, 0]+[next_days_features[feat][day] for feat in features]+([0]*(train_set_scaled.shape[1]-(2+len(features))))
  temp_scaled_features = scaler.transform(np.array([temp_unscaled_features]))[:, 2: 2+len(features)][0]
  for feat_index, feat in enumerate(features):
    scaled_next_days_features[feat].append(temp_scaled_features[feat_index])  

pred_days=6
y_pred_new=[]
for i in range(pred_days):
  #Make the prediction
  y_pred = model.predict(np.array([X_test_new]),batch_size=1)[0][0]
  #Store the prediction
  y_pred_new.append(y_pred)
  #Append the prediction to the previous 'lag1-6' considering it as the 'lag1'
  X_test_new_list = deque(X_test_new[0:-1].tolist())
  X_test_new_list.appendleft([y_pred]+[scaled_next_days_features[feat][i] for feat in features])
  X_test_new = np.array(X_test_new_list) #shape=[timestep, data_dim] 

#Convert the prediction list to the appropriate np array (shape=[num_preds, 1])
y_pred = np.expand_dims(np.array(y_pred_new), axis=1)
#Fill y_pred with dummy zeros for the remaining columns (features)
y_pred_filled = np.append(y_pred, np.zeros((y_pred.shape[0], X_test.shape[1]*X_test.shape[2])), axis=1)
#Apply the inverse transform to the predictions
pred_test_set_inverted = scaler.inverse_transform(y_pred_filled)

#create dataframe that shows the predicted sales
result_list = []
sales_dates = next_dates_list
temp = list(df_sales[-1:].sales)[0]
for index in range(pred_test_set_inverted.shape[0]):
    result_dict = {}
    result_dict['pred_value'] = int(pred_test_set_inverted[index][0] + temp)
    temp = result_dict['pred_value']
    result_dict['date'] = sales_dates[index]
    result_list.append(result_dict.copy())
df_result = pd.DataFrame(result_list)



#####################################################################################

#export the final file with 2 days of pred(the current and the next day)

print(df_result)#['pred_value'])

#df_final = df_result.iloc[:2]

df_final = df_result

df_final.to_excel (r'C:\Python Projects\Everyday_sales_prediction\Exports\pred_values.xlsx', index = False, header=True)


#####################################################################################

#create the file 'history' in order to store historical data of everyday prediction values (6 values/day)

df_result['prediction_date'] = today  #new column with the day which the model run for those pred

history= pd.read_excel(r'C:\Python Projects\Everyday_sales_prediction\Exports\history.xlsx')


history = pd.concat([history, df_result]) #append the new values

history.to_excel (r'C:\Python Projects\Everyday_sales_prediction\Exports\history.xlsx', index = False, header=True)

#####################################################################################

#create the file 'perfomance' in order to store historical data of everyday prediction values and avaluate them with the actual values
#use the last row of data in order to get the last row and the yesterday actual sales 

data.index = np.arange(0, len(data)) #change the index (0-len(data) because due to concat the index had not be continued
history.index = np.arange(0, len(history)) #change the index (0-len(data) because due to concat the index had not be continued

data['Prediction'] = 0 

data['Prediction'][len(data)-1] = history['pred_value'][len(history)-12]


performance= pd.read_excel(r'C:\Python Projects\Everyday_sales_prediction\Exports\performance.xlsx')

performance = pd.concat([performance, data.iloc[-1:]])

performance['dif'] = performance["sales"].sub(performance["Prediction"], axis = 0)

performance['mae'] = (abs(performance["dif"])/performance["sales"])*100

performance = performance.drop(['dif'], axis=1)

performance.to_excel (r'C:\Python Projects\Monthly_prediction_boxes_productivity\Exports', index = False, header=True)

#####################################################################################







#####################################################################################

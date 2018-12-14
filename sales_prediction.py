import sys
sys.path.append('/usr/local/bin')
from numpy.random import seed
seed(1700)
from tensorflow import set_random_seed
set_random_seed(1700)
import csv     
import math
import pandas as pd
import numpy as np
#import matplotlib.pylab as plt
#from matplotlib.pylab import rcParams
#rcParams['figure.figsize'] = 15, 6
from statsmodels.tsa.arima_model import ARIMA
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from math import sqrt
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
#from matplotlib import pyplot
import numpy
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf


product_id_list=[]
product_dict={}
#The program has used LSTM model for final output production

#Model for Arima
def runArimamodel(train,test):
    
    history = [int(item) for item in train]
    predictions = []
    for t in range(len(test)):
        model = ARIMA(history, order=(7,1,0))
        model_fit = model.fit(disp=0)
        res = model_fit.forecast()
        predictedVal = int(res[0][0])
        predictions.append(predictedVal)
        actualVal = test[t]
        history.append(actualVal)
    
    rmseArima = sqrt(mean_squared_error(test, predictions))    
    print('Test MSE: %.3f' % rmseArima)
    
    return history,rmseArima

    
#generate predictions through ARIMA model    
def getPredictionThroughArima(history):
    index=118
    test_predictions=[]
    total_days=(146-118)+1
    for t in range(total_days):
        model = ARIMA(history, order=(7,1,0))
        model_fit = model.fit(disp=0)
        res = model_fit.forecast()
        predicted = int(res[0][0])
        print('%d th Day prediction = %d' %(index,predicted))
        history.append(predicted)
        test_predictions.append(predicted)
        index += 1
    
    return test_predictions


#Convert Timeseries data to supervisedVal data
def convert_timeseries_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


# invert differenced value
def inverse_difference(history, yvalue, interval=1):
    return yvalue + history[-interval]


# normalize train and test data
def normalize(train, test):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    train = train.reshape(train.shape[0], train.shape[1])
    trainsS = scaler.transform(train)
    test = test.reshape(test.shape[0], test.shape[1])
    testS = scaler.transform(test)
    return scaler, trainsS, testS


# inverse scaling for a forecasted value
def inversion_scaled_values(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


# Fot LSTM model
def develop_lstm_model(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True,dropout=0.5,implementation=1, recurrent_dropout=0.5))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)          
        model.reset_states()
    return model


# make a one-step forecast
def get_lstm_predictions(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yvalue = model.predict(X, batch_size=batch_size)
    return yvalue[0,0]



def generate_Predictions_lstm(lstm_model,testS,scaler,orig_values):
   
    #Make predictions on the actual data
    i=118
    test_predictions=[]
    diff=(146-118)+1
    y=testS[len(testS)-1,1:][0]
    
    for t in range(diff):
        X=y
        X = X.reshape(1, 1, 1)
        yvalue = get_lstm_predictions(lstm_model,1,X)
        yList=list()
        yList.append(yvalue)
        y=numpy.array(yList)
        # invert scaling
        yvalue = inversion_scaled_values(scaler, X, yvalue)
        # invert differencing
        yvalue = inverse_difference(orig_values, yvalue, diff+1-t)
        
        val=int(yvalue)
        if(val<0):
            val=0
        print('%d th Day prediction = %f' %(i,val))
        
        #history.append(yvalue)
        test_predictions.append(val)
        i+=1
        
    return test_predictions   


def applylstmmodel(series):
    try:
        # transform data to be stationary
        orig_values = series.values
        diff_values = difference(orig_values, 1)
        
        # transform data to be supervisedVal learning
        supervisedVal = convert_timeseries_supervised(diff_values, 1)
        supervisedValList = supervisedVal.values
        
        lenval=int(len(supervisedValList) * 0.75)
        train, test = supervisedValList[0:lenval], supervisedValList[lenval:]
         
        # transform the normalize of the data
        scaler, trainsS, testS = normalize(train, test)
        
        # fit the model
        lstm_model = develop_lstm_model(trainsS, 1, 900, 5)
        
        # forecast the entire training dataset to build up state for forecasting
        train_RS = trainsS[:, 0].reshape(len(trainsS), 1, 1)
        lstm_model.predict(train_RS, batch_size=1)
        
        # walk-forward validation on the test data
        predictions = list()
        
        for i in range(len(testS)):
    
            # make one-step forecast
            X, y = testS[i, 0:-1], testS[i, -1]
            yvalue = get_lstm_predictions(lstm_model, 1, X)
            # invert scaling
            yvalue = inversion_scaled_values(scaler, X, yvalue)
            # invert differencing
            yvalue = inverse_difference(orig_values, yvalue, len(testS)+1-i)
            # store forecast   
            predictions.append(yvalue)
            expected = orig_values[len(train) + i + 1]
    
        # report performance
        val=len(test)
        rmse = sqrt(mean_squared_error(orig_values[-val:], predictions))
        print('Test RMSE: %.3f' % rmse)
        return rmse,testS,lstm_model,scaler,orig_values
        

    except Exception as e:
        print ("Error in lstm model")
        print  (str(e.args)+"::"+str(e.message))
        
        
        
#Predictions for product_specific sales        
def product_specifc_sales(train_data):
    
    sequence =1
    try:
        with open (train_data, 'r') as f:
            for row in csv.reader(f,delimiter='\t'):  
                #get the each product sales
                product_sales=[int(v) for v in row[1:]]
                product_id=int(row[0])
                product_id_list.append(product_id)   
                product_dict[product_id]=[]
              
                print ("Sequence="+ str(sequence)+"Product Id="+str(product_id))
                sequence += 1
                
                #Generate the time series object
                product_sales_TS = pd.Series((item for item in product_sales))
                series=product_sales_TS   
                rmse,testS,lstm_model,scaler,orig_values= applylstmmodel(series)
                test_predictions= generate_Predictions_lstm(lstm_model, testS,scaler,orig_values)
                product_dict[product_id]=test_predictions
                
                
    except Exception as e:
        print "Error in opening Training File"
        print str(e.message)+"::"+str(e.args)
       


#Predictions for overall sales data
def overall_sales(train_data):
    datalist=[]
    try:
        with open (train_data, 'r') as f:
            for row in csv.reader(f,delimiter='\t'):
                datalist.append(row)

    except Exception as e:
        print str(e.message)+"::"+str(e.args)
        
    
    overall_sales=[]
    overall_sales_day=[]
    
    for item in datalist:
        sales=[int(v) for v in item[1:]]
        overall_sales_day.append(sales)
        
    overall_sales=[sum(i) for i in zip(*overall_sales_day)]
    overall_sales_TS = pd.Series((v for v in overall_sales))
    
    product_id=0
    product_id_list.append(product_id)

    series=overall_sales_TS
    rmse,testS,lstm_model,scaler,orig_values= applylstmmodel(series)
    test_predictions= generate_Predictions_lstm(lstm_model, testS,scaler,orig_values)
    product_dict[product_id]=test_predictions

            
    
def main():

    #Get the Training Data
    if len(sys.argv) < 2:
        print("Input directory path is required")
        print("Execution format: python <program_name.py> <inputfile.txt>")
        sys.exit(1)

    else:
        train_data = str(sys.argv[1])
        
    reload(sys)
    sys.setdefaultencoding("utf-8")           
 
    #generte Prediction for overall Sales Data
    overall_sales(train_data)
 
    #Generate Prediction for product specific sales
    product_specifc_sales(train_data)
    
    #Write Output to output.txt
    try:
        with open('output.txt', 'w') as f:
            for item in product_id_list:
                key=item
                val=product_dict[key]
                f.write(str(key)+"\t")
                for v in val:
                    f.write(str(int(v)) + "\t")        
                f.write("\n")    

    except Exception as e:
        print ("Error in writing file")
        print (str(e.message)+"::"+str(e.args))
        
if __name__ == "__main__":
    main()
    
    
    
    
    
    


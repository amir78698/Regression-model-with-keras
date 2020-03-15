"""
Develope a Regrssion model with Keras for prediction of values
"""


# importing required libraries
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense



"""
for preprocesing of Dataset
"""

# read data into pandas dataframe
required_data = pd.read_csv('put the address of your data')


# look at the dimensions of the data
required_data.shape

# split dataset into pridictors and target
required_data_columns = required_data.columns
predictors = required_data[required_data_columns[required_data_columns != 'put the column name of your target ']] # all columns except Target
target = required_data['Target Column name'] # Strength column


#step is to normalize the data by substracting the mean and dividing by the standard deviation.
predictors_norm = (predictors - predictors.mean()) / predictors.std() 

#Let's save the number of predictors to n_cols since we will need this number when building our network.
n_cols = predictors_norm.shape[1] # number of predictors

""" Building a neural network"""

# define regression model
def regression_model():
    # create model
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# build the model
model = regression_model()

# fit the model
model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)




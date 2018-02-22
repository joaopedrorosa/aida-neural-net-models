import numpy
import pandas as pd
dataframe = pd.read_csv("tese-regression-model-v1\dataset_final.csv")
dataset = dataframe.values

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

X = dataset[:,35:39]
y = dataset[:,0:35]

poly = PolynomialFeatures(2)
X_poly = poly.fit_transform(X) #we now have a feature vector with 15 rows instead of only 4

standard_scaler = StandardScaler()
x_scaled = standard_scaler.fit_transform(X_poly)

scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y)

import keras.backend as K
import tensorflow as tf

def custom_loss(y_true, y_pred):
    
    percentage = 0.3
    
    pred_class = y_pred[..., 0:2]
    true_class = tf.argmax(y_true[..., 0:2], -1)
    loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_class, logits=pred_class)
    
    loss_reg1 = K.mean(K.square(y_pred[...,2:17] - y_true[...,2:17]), axis=-1)
    loss_reg2 = K.mean(K.square(y_pred[...,17:35] - y_true[...,17:35]), axis=-1)
    
    loss = percentage*loss_class + (1-percentage)*(loss_reg1*y_true[..., 0] + loss_reg2*y_true[..., 1])
    
    return loss

import keras
from keras.models import Sequential
from keras.layers import Dense    
from keras import regularizers

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(15, input_dim=15, kernel_initializer='normal',  activation='relu'))
    model.add(Dense(35, kernel_initializer='normal'))
    # Compile model
    model.compile(loss=custom_loss, optimizer='adam') 
    
    return model

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime

X_train, X_test, y_train, y_test = train_test_split(x_scaled, y_scaled,
                                                    test_size = 0.2,
                                                    random_state = seed)

model = baseline_model()

#Starting Time
start_time = datetime.now()
print('Start Time:', start_time)

history = model.fit(x_scaled, 
                    y_scaled, 
                    validation_split = 0.33,
                    epochs = 500, 
                    batch_size= 128, 
                    verbose = 0)

stop_time = datetime.now()

elapsed_time = stop_time - start_time
print('Elapsed Time:', elapsed_time) 

y_proba = model.predict(x_scaled)
predicted_inverse = scaler.inverse_transform(y_proba)
print(y[1789])
print(predicted_inverse[1789])
print(y_proba[1789])
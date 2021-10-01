#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 08:58:47 2021

@author: jinshengdan

"""


###  ---- Code ANN regression using KERAS, train on the Boston housing dataset ---- ###

## Import libraries
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
## Load in data
from keras.datasets import boston_housing
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

## Check the data
train_data.shape #(404,13)
test_data.shape #(102,13)

train_targets # the median values of owner-occupied homes, in thousands of dollars

## Normalize the data
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

## Build Model

def build_model():
    
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

## K-fold validation

k=4
num_val_samples = len(train_data) // k 
#num_epochs = 100
#all_scores = []
'''
for i in range(k):
    print('processing fold #', i)
    # prepare the validation data - data from partitopm #k
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples] 
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    # prepare the training data - data from all other partitions
    partial_train_data = np.concatenate( 
            [train_data[:i * num_val_samples],
             train_data[(i + 1) * num_val_samples:]], 
             axis=0)
    partial_train_targets = np.concatenate( 
            [train_targets[:i * num_val_samples],
             train_targets[(i + 1) * num_val_samples:]], 
             axis=0)
    # builds the keras model
    model = build_model()
    # train the model
    model.fit(partial_train_data, partial_train_targets,
              epochs=num_epochs, batch_size=1, verbose=0)
    # evaluate the model on the validation data
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)
    
      
all_scores # results from 4 folds with 100 epochs
np.mean(all_scores)

'''

## Save the validation logs at each fold
num_epochs = 500 
all_mae_histories = [] 
for i in range(k):
    print('processing fold #', i)
    # prepare the validation data - data from partition #k
 
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples] 
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    # prepare the training data - data from other partitions
    partial_train_data = np.concatenate( 
            [train_data[:i * num_val_samples],
             train_data[(i + 1) * num_val_samples:]], 
             axis=0)

    partial_train_targets = np.concatenate( 
            [train_targets[:i * num_val_samples],
             train_targets[(i + 1) * num_val_samples:]], 
             axis=0)
    # build the keras model 
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0) 
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)

## Building the history of successive mean K-fold validation scores
average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

## Plotting validation scores
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

## Plotting validation scores, excluding the first 10 data points
def smooth_curve(points, factor=0.9):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point)
  return smoothed_points
smooth_mae_history = smooth_curve(average_mae_history[10:])
plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

## Training the final model
model = build_model()
# train on the entirety of the data
model.fit(train_data, train_targets, epochs=80, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
# result
test_mae_score



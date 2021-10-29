#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 20:40:35 2021

@author: jinshengdan
"""
from sklearn.model_selection import train_test_split
batch_size = 32

### Get Models from second script
import importlib  
models = importlib.import_module("02-train")

_1D_CNN = models._1D_CNN()
RNN = models.RNN()

## import 01-clean.py
## Get train, test, validation dataset
clean_ = __import__('01-clean')
## Load processed data from 01-clean.py
x_train = clean_.load_xtrain()
y_train = clean_.load_ytrain()
x_test = clean_.load_xtest()
y_test=clean_.load_ytest()

x_train.shape # (120,100)
y_train.shape # (120,)
x_test.shape # (33, 100)
y_test.shape # (33,)

x_train=x_train.reshape(120,100,1)
y_train=y_train.reshape(120,1,1)
x_test=x_test.reshape(33,100,1)
y_test=y_test.reshape(33,1,1)

x_train,x_valid,y_train,y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=13)


### 1D CNN

_1D_CNN.load_weights('Model_1D-CNN.h5')

train_loss, train_acc = _1D_CNN.evaluate(x_train, y_train, batch_size=batch_size)
valid_loss, valid_acc = _1D_CNN.evaluate(x_valid, y_valid, batch_size=batch_size)
test_loss, test_acc = _1D_CNN.evaluate(x_test, y_test, batch_size=x_test.shape[0])

print('train_acc:', train_acc)
print('valid_acc:', valid_acc)
print('test_acc:', test_acc)
loss, accuracy = _1D_CNN.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)


### RNN

RNN.load_weights('Model_RNN.h5')

train_loss, train_acc = RNN.evaluate(x_train, y_train, batch_size=batch_size)
valid_loss, valid_acc = RNN.evaluate(x_valid, y_valid, batch_size=batch_size)
test_loss, test_acc = RNN.evaluate(x_test, y_test, batch_size=x_test.shape[0])

print('train_acc:', train_acc)
print('valid_acc:', valid_acc)
print('test_acc:', test_acc)
loss, accuracy = RNN.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)

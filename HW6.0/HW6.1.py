#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 19:28:14 2021

@author: jinshengdan
"""

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from keras import models
from keras import layers

#GET DATASET
from keras.datasets import mnist
(X, Y), (test_images, test_labels) = mnist.load_data()

#NORMALIZE AND RESHAPE
X=X/np.max(X) 
test_images=test_images/np.max(test_images) 
X=X.reshape(60000,28*28)
test_images=test_images.reshape(10000,28*28); 
 
#MODEL
n_bottleneck=90

# SHALLOW
model = models.Sequential()
model.add(layers.Dense(28*28,  activation='linear',input_shape=(28 * 28,)))
model.add(layers.Dense(n_bottleneck, activation='linear'))
model.add(layers.Dense(28*28,  activation='linear'))


batch_size = 1000
epochs = 10

#COMPILE AND FIT
model.compile(optimizer='rmsprop',
                loss='mean_squared_error',metrics='acc')
model.summary()
#model.fit(X, X, epochs=10, batch_size=1000,validation_split=0.2)

history = model.fit(X, X, epochs=epochs, batch_size=batch_size,validation_split=0.2)
loss, accuracy = model.evaluate(test_images, test_labels, batch_size=batch_size, verbose=0)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


#EXTRACT MIDDLE LAYER (REDUCED REPRESENTATION)
from keras import Model 
extract = Model(model.inputs, model.layers[-2].output) # Dense(128,...)
X1 = extract.predict(X)
print(X1.shape)


#2D PLOT
plt.scatter(X1[:,0], X1[:,1], c=Y, cmap='tab10')
plt.show()

#3D PLOT
ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=X1[:,0], 
    ys=X1[:,1], 
    zs=X1[:,2], 
    c=Y, 
    cmap='tab10'
)
plt.show()



#PLOT ORIGINAL AND RECONSTRUCTED 
X1=model.predict(X) 

test_loss, test_acc = model.evaluate(X, Y, batch_size=X.shape[0])
print('test_acc:', test_acc)

#RESHAPE
X=X.reshape(60000,28,28); #print(X[0])
X1=X1.reshape(60000,28,28); #print(X[0])
#test_images=test_images.reshape(10000,28,28)

#COMPARE ORIGINAL 
f, ax = plt.subplots(4,1)
I1=11; I2=46
ax[0].imshow(X[I1])
ax[1].imshow(X1[I1])
ax[2].imshow(X[I2])
ax[3].imshow(X1[I2])
plt.show()


####  Use this trained AE to perform anomaly detection using the MNIST-FASHION data (as anomalies)

from keras.datasets import fashion_mnist
(X_, Y_), (test_images_, test_labels_)  = fashion_mnist.load_data()

#NORMALIZE AND RESHAPE
X_=X_/np.max(X_) 
test_images_=test_images_/np.max(test_images_) 
X_=X_.reshape(60000,28*28)
test_images_=test_images_.reshape(10000,28*28); 

'''
X1_ = extract.predict(X_)
print(X1_.shape)

#2D PLOT
plt.scatter(X1[:,0], X1[:,1], c=Y, cmap='tab10')
plt.show()

#3D PLOT
ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=X1[:,0], 
    ys=X1[:,1], 
    zs=X1[:,2], 
    c=Y, 
    cmap='tab10'
)
plt.show()

#PLOT ORIGINAL AND RECONSTRUCTED 
X1_=model.predict(X_) 

#RESHAPE
X_=X_.reshape(60000,28,28); #print(X[0])
X1_=X1_.reshape(60000,28,28); #print(X[0])
#test_images=test_images.reshape(10000,28,28)

'''

#########################

X=np.asarray(X)
Y=np.asarray(Y)
test_images=np.asarray(test_images)
test_labels=np.asarray(test_labels)
# Rescale the image data to 0 ~ 1.
X = X.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0
X = X.reshape((len(X), np.prod(X.shape[1:])))
test_images = test_images.reshape((len(test_images), np.prod(test_images.shape[1:])))

X_=np.asarray(X_)
Y_=np.asarray(Y_)
X_ = X_.astype(np.float32) / 255.0
X_ = X_.reshape((len(X_), np.prod(X.shape[1:])))

set_num = 50
x_sorted = test_images[test_labels == 0]
x_sorted = x_sorted[:set_num]
for i in range(1,10):
    x = test_images[test_labels == i]
    x_sorted = np.concatenate([x_sorted, x[:set_num]], axis=0)
x_sorted = np.concatenate([x_sorted, X_[:][:set_num]], axis=0)


y_sorted = test_labels[test_labels == 0]
y_sorted = y_sorted[:set_num]
for i in range(1,20):
    y = test_labels[test_labels == i]
    y_sorted = np.concatenate([y_sorted, y[:set_num]], axis=0)
y_sorted = np.concatenate([y_sorted, Y_[:][:set_num]], axis=0)


fashion_label = 10


#PLOT ORIGINAL AND RECONSTRUCTED 
X1_=model.predict(x_sorted) 


test_loss, test_acc = model.evaluate(x_sorted, y_sorted, batch_size=x_sorted.shape[0])
print(' fraction of times anomalies are detected', test_acc)

#RESHAPE
x_sorted = x_sorted.reshape(550,28,28)
X1_=X1_.reshape(550,28,28)


#COMPARE ORIGINAL 
f, ax = plt.subplots(4,2)
I1=32; I2=400; I3=500;I4=510
ax[0,0].imshow(x_sorted[I1])
ax[0,1].imshow(X1_[I1])
ax[1,0].imshow(x_sorted[I2])
ax[1,1].imshow(X1_[I2])
ax[2,0].imshow(x_sorted[I3])
ax[2,1].imshow(X1_[I3])
ax[3,0].imshow(x_sorted[I4])
ax[3,1].imshow(X1_[I4])

plt.show()

### Save Models          
model.save("Model_AE_MNIST.h5")

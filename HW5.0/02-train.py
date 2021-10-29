#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 17:12:40 2021

@author: jinshengdan
"""

from keras import layers 
from keras import models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

for i in ['1D-CNN','RNN']:
    Model = i
    verbose = 0
    epochs = 10
    batch_size =32
    
    ## import 01-clean.py
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
    
    #x_train_o = x_train
    #y_train_o = y_train
    #x_test_o = x_test
    #y_test_o = y_test
    
    x_train=x_train.reshape(120,100,1)
    y_train=y_train.reshape(120,1,1)
    x_test=x_test.reshape(33,100,1)
    y_test=y_test.reshape(33,1,1)
    
    x_train,x_valid,y_train,y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=13)
    n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
    
    
    def _1D_CNN():
        # fit and evaluate a model    
        model = models.Sequential()
        model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(100,1)))
        model.add(layers.Dropout(0.5))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Flatten())
        model.add(layers.Dense(100, activation='relu'))
        model.add(layers.Dense(n_outputs, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
        # fit network
    def RNN():
        model = models.Sequential() 
        model.add(layers.Embedding(10000, 32))
        model.add(layers.SimpleRNN(32, return_sequences=True))
        model.add(layers.SimpleRNN(32, return_sequences=True))
        model.add(layers.SimpleRNN(32, return_sequences=True))
        model.add(layers.SimpleRNN(32))
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc']) 
        return model
     
    
    
    ## 1D CNN with Dropout
    if (Model == '1D-CNN'):
        model = _1D_CNN()
        
        model_train=model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose,validation_split=0.2)
        model.summary()
        # evaluate model
            
        train_loss, train_acc = model.evaluate(x_train, y_train, batch_size=batch_size)
        valid_loss, valid_acc = model.evaluate(x_valid, y_valid, batch_size=batch_size)
        test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=x_test.shape[0])
        
        print('train_acc:', train_acc)
        print('valid_acc:', valid_acc)
        print('test_acc:', test_acc)
        loss, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
        
        accuracy = model_train.history['accuracy']
        val_accuracy = model_train.history['val_accuracy']
        loss = model_train.history['loss']
        val_loss = model_train.history['val_loss']
        epochs = range(len(accuracy))
        plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
        plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy'+' - '+Model)
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss'+' - '+Model)
        plt.legend()
        plt.show()
        
        '''
        from sklearn.metrics import roc_aus_score
        from sklearn.metrics import roc_curve
        
        FPR, TPR, thresholds = roc_curve(y_test)
        
        
        from sklearn.metrics import roc_curve
        model_t=model.fit(x_train,y_train)
        preds = model.predict(x_test)
        #preds=preds.reshape(33,1,1)
        true_labels = y_test
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import accuracy_score
        confusion_matrix(true_labels, preds)
        accuracy_score(true_labels, preds)
        #fpr, tpr, threshold = roc_curve(true_labels, preds)
        '''
        
        
    if (Model == 'RNN'):
        model = RNN()
    
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.2)
        model.summary()
        train_loss, train_acc = model.evaluate(x_train, y_train, batch_size=batch_size)
        valid_loss, valid_acc = model.evaluate(x_valid, y_valid, batch_size=batch_size)
        test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=x_test.shape[0])
        
        print('train_acc:', train_acc)
        print('valid_acc:', valid_acc)
        print('test_acc:', test_acc)
        loss, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
        
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy'+' - '+Model)
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss'+' - '+Model)
        plt.legend()
        plt.show()
        
        
### Save Models          
model.save("Model_"+Model+".h5")    
    

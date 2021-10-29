#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 16:03:12 2021

@author: jinshengdan
"""

import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np



train_dir = os.getcwd()+'/train'
labels = []
texts = []
for label_type in ['AUGUSTUS','BENTONS VENTURE','Pride and Prejudice']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
            if label_type == 'AUGUSTUS': # label 0 
                labels.append(0)
            elif label_type == 'BENTONS VENTURE': # label 1
                labels.append(1)
            else: # label 2
                labels.append(2)
#len(texts) # 153
#labels # 0,1,2
unique_words = set(texts)
print(len(unique_words))

maxlen = 100  # Cuts off reviews after 100 words
training_samples = 120  # Trains on 120 samples
validation_samples = 33 # Validates on 33 samples 
max_words = 1000 #Considers only the top 10,000 words in the dataset

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=maxlen)
labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_test = data[training_samples: training_samples + validation_samples] 
y_test = labels[training_samples: training_samples + validation_samples]

## Build functions to save and return dataset for later

def load_xtrain():
    return x_train

def load_ytrain():
    return y_train

def load_xtest():
    return x_test

def load_ytest():
    return y_test
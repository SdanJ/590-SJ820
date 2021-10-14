#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 13:46:34 2021

@author: jinshengdan
"""

##########################
### HW 4.0 Part 1
### ANLY 590 Section 2
### Shengdan Jin
##########################

#MODIFIED FROM CHOLLETT P120 
from keras import layers 
from keras import models
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
#from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
warnings.filterwarnings("ignore")

#-------------------------------------
# ORGANIZE ALL HYPER PATAMETERS
#-------------------------------------
for i in ['MNIST','MNIST FASHION', 'CIFAR-10']:
    DATASET = i
    #DATASET = 'MNIST FASHION'
    model_type = "CNN"
    activation = "SIGMOID"
    DATA_AUG = True
    NKEEP=10000
    batch_size=int(0.05*NKEEP)
    epochs=20
    
    
    #-------------------------------------
    #GET DATA AND REFORMAT
    #-------------------------------------
    
    ### MNIST
    if (DATASET == 'MNIST'):
        from keras.datasets import mnist
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        #print(train_images.shape,test_images.shape) # (60000, 28, 28) (10000, 28, 28)
        train_images = train_images.reshape((60000, 28, 28, 1))
        test_images = test_images.reshape((10000, 28, 28, 1))
    
    
    ### MNIST Fashion
    if (DATASET == 'MNIST FASHION'):
        from keras.datasets import fashion_mnist
        (train_images, train_labels), (test_images, test_labels)  = fashion_mnist.load_data()
        # print(train_images.shape,test_images.shape) # (60000, 28, 28) (10000, 28, 28)
        train_images = train_images.reshape((60000, 28, 28, 1))
        test_images = test_images.reshape((10000, 28, 28, 1))
    
    ### CIFAR-10
    if (DATASET == 'CIFAR-10'):
        from keras.datasets import cifar10
        (train_images, train_labels), (test_images, test_labels)  = cifar10.load_data()
        #print(train_images.shape,test_images.shape) # (50000, 32, 32, 3) (10000, 32, 32, 3)
    
    
    #DEBUGGING
    print("batch_size",batch_size)
    rand_indices = np.random.permutation(train_images.shape[0])
    train_images=train_images[rand_indices[0:NKEEP],:,:]
    train_labels=train_labels[rand_indices[0:NKEEP]]
    # exit()
    
    #NORMALIZE
    train_images = train_images.astype('float32') / 255 
    test_images = test_images.astype('float32') / 255  
    
    
    
    #CONVERTS A CLASS VECTOR (INTEGERS) TO BINARY CLASS MATRIX.
    tmp=train_labels[0]
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    print(tmp, '-->',train_labels[0])
    print("train_labels shape:", train_labels.shape)
    
    #----------------------------------------------------------
    # 80-20 Split of Training set into train and validation set
    #---------------------------------------------------------- 
    train_X,valid_X,train_label,valid_label = train_test_split(train_images, train_labels, test_size=0.2, random_state=13)
    
    
    #----------------------------------------
    # Plot of the first image of training set
    #----------------------------------------
    
    plt.imshow(train_X[0,:,:], cmap='gray')
    plt.title("The First Image of Dataset - "+DATASET)
    plt.show()
    
    #----------------------------------------
    # DATA AUGMENTATION
    #----------------------------------------
    
    if (DATA_AUG == True):
        x=train_X[0,:,:]
        x = x.reshape((1,) + x.shape)
        ## Data Augmentation
        datagen = ImageDataGenerator(
              rotation_range=40,
              width_shift_range=0.2,
              height_shift_range=0.2,
              shear_range=0.2,
              zoom_range=0.2,
              horizontal_flip=True,
              fill_mode='nearest')
        
        i=0
        for batch in datagen.flow(x, batch_size=1):
            plt.figure(i)
            imgplot = plt.imshow(image.array_to_img(batch[0]))
            plt.title("Augmentation for the First Image of Dataset_"+str(i+1)+" - "+DATASET)
            i += 1
            if i % 4 == 0:
                break
        
        plt.show()
    
    #-------------------------------------
    #BUILD MODEL SEQUENTIALLY (LINEAR STACK)
    #-------------------------------------
    if (model_type=="CNN"):
        model = models.Sequential()
        
        if (DATASET == 'CIFAR-10'):
            model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        else:
            model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        
        model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))
        
        model.summary()
        
        #-------------------------------------
        #COMPILE AND TRAIN MODEL
        #-------------------------------------
            
        model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
        
        model_train = model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))
    
        #-------------------------------------
        # PLOT Train/Validation History
        #-------------------------------------
    
        accuracy = model_train.history['accuracy']
        val_accuracy = model_train.history['val_accuracy']
        loss = model_train.history['loss']
        val_loss = model_train.history['val_loss']
        epochs = range(len(accuracy))
        plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
        plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy - '+DATASET)
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss - '+DATASET)
        plt.legend()
        plt.show()
        
        
    #-------------------------------------
    #EVALUATE ON TEST DATA
    #-------------------------------------
    train_loss, train_acc = model.evaluate(train_X, train_label, batch_size=batch_size)
    valid_loss, valid_acc = model.evaluate(valid_X, valid_label, batch_size=batch_size)
    test_loss, test_acc = model.evaluate(test_images, test_labels, batch_size=test_images.shape[0])
    
    print('train_acc:', train_acc)
    print('valid_acc:', valid_acc)
    print('test_acc:', test_acc)
    
    
    #---------------------------------------
    # Visualize the intermediate activations 
    #---------------------------------------
    layer_outputs = [layer.output for layer in model.layers[:5]] 
    x= test_images[0,:,:] #test data
    x =x.reshape((1,) + x.shape)
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(x)
    first_layer_activation = activations[0]
    ## Visualizing the 10th channel
    #plt.matshow(first_layer_activation[0, :, :, 10], cmap='viridis')
    
    layer_names = []
    for layer in model.layers[:5]:
        layer_names.append(layer.name)
    images_per_row = 16
    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0,:, :,col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size,
                 row * size : (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
        plt.title(layer_name+' - '+DATASET)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()
    
    
    #-------------------------------------
    # SAVE MODEL AND HYPER PARAMETERS
    #-------------------------------------
    model.save("Model_"+DATASET+".py")
    
    #-------------------------------------
    # READ A MODEL FROM A FILE
    #-------------------------------------
    #from keras.models import load_model
    #model = load_model('Model_'+DATASET+".py")
    #model.summary()

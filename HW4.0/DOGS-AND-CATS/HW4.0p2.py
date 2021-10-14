#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 17:00:17 2021

@author: jinshengdan
"""
##########################
### HW 4.0 Part 2
### ANLY 590 Section 2
### Shengdan Jin
##########################

import os, shutil
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

# Path to the directory where the original dataset was uncompressed
original_dataset_dir = '/Users/jinshengdan/590-SJ820/HW4.0/DOGS-AND-CATS/dogs-vs-cats'
# Directory where the smaller dataset will be stored
base_dir = '/Users/jinshengdan/590-SJ820/HW4.0/DOGS-AND-CATS/cats_and_dogs_small'
os.mkdir(base_dir)

# Directories for the training, validation, and test splits
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

# Directory with traininng cat pictures
train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)
# Directory with training dog pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)
# Directory with validation cat pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)
# Directory with validation dog pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)
# Directory with test cat pictures
test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)
# Directory with test dog pictures
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)

# Copies the first 1,000 cat images to train_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)
# Copies the next 500 cat images to validation_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)
# Copies the next 500 cat images to test_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)
# Copies the first 1,000 dog images to train_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)
# Copies the next 500 dog images to validation_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)
# Copies the next 500 dog images to test_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)

## Check how many pictures are in each training split (train/vali- dation/test)
print('total training cat images:', len(os.listdir(train_cats_dir))) # 1000
print('total training dog images:', len(os.listdir(train_dogs_dir))) # 1000
print('total validation cat images:', len(os.listdir(validation_cats_dir))) # 500
print('total validation dog images:', len(os.listdir(validation_dogs_dir))) # 500
print('total test cat images:', len(os.listdir(test_cats_dir))) # 500
print('total test dog images:', len(os.listdir(test_dogs_dir))) # 500



## Build Model
# Instantiating a small convnet for dogs vs. cats classification
from keras import layers
from keras import models
model = models.Sequential()

# Add layers from different dimensions and different activation functions
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Check the dimensions of the feature maps change with every successive layer
model.summary()

# Configuring the model for training
from tensorflow.keras import optimizers
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# Data Processing
# Using ImageDataGenerator to read images from directories
from keras.preprocessing.image import ImageDataGenerator
# Rescales all images by 1/255
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir, # target directory
        target_size=(150,150), # resizes all images to 150x150
        batch_size=20,
        class_mode='binary') # Binary labels needed because binary_crossentropy loss is used.

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')
# Check the output of one of these generators
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape) # it yields batches of 150 × 150 RGB images (shape (20, 150, 150, 3)) 
    print('labels batch shape:', labels_batch.shape) # binary labels (shape (20,))
    break

# Fit the model
history = model.fit_generator(
      train_generator,
      steps_per_epoch=50,
      epochs=5,
      validation_data=validation_generator,
      validation_steps=50)
# Save the model
model.save('cats_and_dogs_small_1.h5')
# Plot the loss and accuracy of the model over the training and validation data during training
import matplotlib.pyplot as plt
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

# Using data augmentation
# Setting up a data augmentation configuration via ImageDataGenerator
datagen = ImageDataGenerator(
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
# Displaying some randomly augmented training images
from keras.preprocessing import image # Module with image preprocessing utilities
fnames = [os.path.join(train_cats_dir, fname) for
     fname in os.listdir(train_cats_dir)]
img_path = fnames[3] # Chooses one image to augment
img = image.load_img(img_path, target_size=(150, 150)) # Reads the image and resizes it
x = image.img_to_array(img) # Converts it to a Numpy array with shape (150, 150, 3)
x = x.reshape((1,) + x.shape) # Reshapes it to (1, 150, 150, 3)
# Generates batches of randomly transformed images
# Loops indefinitely
i=0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
plt.show()

# Defining a new convnet that includes dropout
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])


# Training the convnet using data-augmentation generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir, # target directory
        target_size=(150, 150), # resizes all images to 150x150
        batch_size=32,
        class_mode='binary') # Binary labels needed because binary_crossentropy loss is used.

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

history = model.fit_generator(
      train_generator,
      steps_per_epoch=1,
      epochs=50,
      validation_data=validation_generator,
      validation_steps=50)

# Save the model
model.save('cats_and_dogs_small_2.h5')

# Visualizing what convnets learn
# Visualizing intermediate activations
from keras.models import load_model
model = load_model('cats_and_dogs_small_2.h5')
model.summary()

# Preprocessing a single image
img_path = '/Users/jinshengdan/590-SJ820/HW4.0/DOGS-AND-CATS/cats_and_dogs_small/test/cats/cat.1700.jpg'
from keras.preprocessing import image  # Preprocesses the image into a 4D tensor
import numpy as np
img = image.load_img(img_path, target_size=(150, 150)) 
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0) 
img_tensor /= 255
# print the shape
print(img_tensor.shape) #(1, 150, 150, 3)

# Displaying the test picture
import matplotlib.pyplot as plt
plt.imshow(img_tensor[0])
plt.show()

# Instantiating a model from an input tensor and a list of output tensors
from keras import models
# Extracts the outputs of the top eight layers
layer_outputs = [layer.output for layer in model.layers[:8]] 
# Creates a model that will return these outputs, given the model input
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

# Running the model in predict mode
# Returns a list of five Numpy arrays: one array per layer activation
activations = activation_model.predict(img_tensor)
# the activation of the first convolution layer for the cat image input
first_layer_activation = activations[0]
print(first_layer_activation.shape) # check the shape

# Visualizing the fourth channel
#import matplotlib.pyplot as plt
#plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
# Visualizing the seventh channel
#plt.matshow(first_layer_activation[0, :, :, 7], cmap='viridis')

# Visualizing every channel in every intermediate activation 
layer_names = []
for layer in model.layers[:8]:
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
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()
    


#!/usr/bin/env python
# coding: utf-8

# ## Fine tune VGG Model 

import sys
from numpy import load
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from keras import backend
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.optimizers import SGD
from keras.applications.vgg16 import VGG16
from keras.models import Model

#define loss function for multi-lable classification 
def fbeta(y_true, y_pred, beta=2):
    y_pred = backend.clip(y_pred, 0, 1)
    tp = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)), axis=1)
    fp = backend.sum(backend.round(backend.clip(y_pred - y_true, 0, 1)), axis=1)
    fn = backend.sum(backend.round(backend.clip(y_true - y_pred, 0, 1)), axis=1)
    p = tp / (tp + fp + backend.epsilon())
    r = tp / (tp + fn + backend.epsilon())
    bb = beta ** 2
    fbeta_score = backend.mean((1 + bb) * (p * r) / (bb * p + r + backend.epsilon()))
    return fbeta_score
   

def define_modelVGG(in_shape=(150, 150, 3), out_shape = 40):     ## class number 
    model = VGG16(include_top = False, weights='imagenet', input_shape = in_shape)
    
    for layer in model.layers:
        layer.trainable = False     
        
    model.get_layer('block5_conv1').trainable = True
    model.get_layer('block5_conv2').trainable = True
    model.get_layer('block5_conv3').trainable = True
    model.get_layer('block5_pool').trainable = True
    
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation = 'relu', kernel_initializer = 'he_uniform')(flat1)
    output = Dense(out_shape, activation='sigmoid')(class1)
    model = Model(inputs = model.inputs, outputs = output)
    
    opt = SGD(lr = 0.005, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[fbeta])
    return model

model = define_modelVGG()
history = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size = 16, epochs = 30, verbose=1)                              
loss, fbeta = model.evaluate(testX, testY)   
print('> loss=%.3f, fbeta=%.3f' % (loss, fbeta))
model.save('final_antiques_model.h5')        


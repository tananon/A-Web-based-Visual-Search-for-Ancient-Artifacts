#!/usr/bin/env python
# coding: utf-8

# ## Input and Output 

# In[ ]:


from pandas import read_csv
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from os import listdir
from sklearn.model_selection import train_test_split
from numpy import zeros,asarray

folder = '/thumb/'

def create_label_mapping(mapping_csv):
    labels = set()    
    for i in range(len(mapping_csv)):
        x = [mapping_csv['antique_type_id'][i]]
        y = [mapping_csv['style_id'][i]]        
        labels.update(x)              
        labels.update(y)                  
    labels = list(labels)        
    labels.sort()    
    labels_map = {labels[i]:i for i in range(len(labels))}
    inv_labels_map = {i:labels[i] for i in range(len(labels))}
    return labels_map, inv_labels_map

def create_file_mapping(mapping_csv):
    mapping = dict()    
    for i in range(len(mapping_csv)):
        name = mapping_csv['file_name'][i] 
        type_name = mapping_csv['antique_type_id'][i]        
        style = mapping_csv['style_id'][i]       
        mapping[name] = [type_name, style]
    return mapping

# create a one hot encoding for one list of tags
def one_hot_encode(tags, mapping):
    encoding = zeros(len(mapping), dtype='uint8')
    for tag in tags:
        encoding[mapping[tag]] = 1
    return encoding

# load all images into memory
def load_dataset(path, file_mapping, tag_mapping):
    photos, targets = list(), list()
    for filename in listdir(folder):        
        if filename in file_mapping:             
            photo = load_img(path + filename, target_size = (150, 150))
            photo = img_to_array(photo, dtype = 'uint8')
            labels = file_mapping[filename]
            target = one_hot_encode(labels, tag_mapping)
            photos.append(photo)
            targets.append(target)
    X = asarray(photos, dtype='uint8')
    y = asarray(targets, dtype='uint8')   
    return X, y
    
filename = 'antiquitiess.csv'
label_mapping, _ = create_label_mapping(df)
file_mapping = create_file_mapping(df)
print('...... load_data ......')
X, y = load_dataset(folder, file_mapping, label_mapping)

print('read image data  150x150')
trainX, testX, trainY, testY = train_test_split(X, y, test_size = 0.2, random_state = 1)
print(trainX.shape, trainY.shape, testX.shape, testY.shape)


# ## Fine tune VGG Model 

# In[ ]:


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


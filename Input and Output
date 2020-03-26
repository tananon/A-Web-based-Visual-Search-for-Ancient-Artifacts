#!/usr/bin/env python
# coding: utf-8

# ## Input and Output 

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

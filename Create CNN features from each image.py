import numpy as np
import os
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model, Model
from os import listdir
from annoy import AnnoyIndex
from keras import backend
import mysql.connector


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='final_antiques_model.h5', type=str,
                    help='weight model')
parser.add_argument('--index', default='antiques_idx.ann', type=str,
                    help='The filename of image index')
parser.add_argument('--path', default='thumb/', type=str,
                    help='path of images')


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


def load_database(user1 ='root', password1 = 'b12345678b', host1 = '202.44.40.189', database1 ='antique'):    
    con = mysql.connector.connect(user = user1, password = password1, host= host1, database= database1, auth_plugin='mysql_native_password')
    cursor = con.cursor()
    sql = ("select file_name, id from files where file_name is not null")
    cursor.execute(sql)
    p = cursor.fetchall()
    data = dict(p)
    return data

def load_image(filename):
    # load the image        
    img = load_img(filename, target_size = (150, 150))    
    img = img_to_array(img)    
    img = img.reshape(1, 150, 150, 3)    
    img = img.astype('uint8') 
    return img

def normalize(x):    
    normalized_values = list()
    maximum = np.max(x)
    minimum = np.min(x)
    for y in x:        
        x_normalized = ( y - minimum) / (maximum - minimum)
        normalized_values.append(x_normalized)
    n_array = np.array(normalized_values)
    return n_array

def build_index(VEC_LENGTH = 128, NUM_TREES = 30):                  ### Extract features from VGG    
    
    model = load_model(args.model, custom_objects={'fbeta': fbeta})    ### load .h5 model 
    
    intermediate_layer_model = Model(input = model.input, output = model.get_layer('dense_1').output)
    print('load model is okay')
    
    t = AnnoyIndex(VEC_LENGTH, 'angular')                         ### length 128 
    folder = args.path    
    db = load_database()                                          ### load database     
    print('load database is okay')
    
    for filename in listdir(folder):
        if filename in db.keys():
            print('load ', filename)
            img = load_image(os.path.join(folder, filename))      ### load img
            feature = intermediate_layer_model.predict(img)       ### get feature from img
            data = np.reshape(feature, (-1 , ))             
            x = normalize(data)
            t.add_item(db[filename], x)                           ## add index 
        
    t.build(NUM_TREES)
    t.save(args.index)

    
if __name__ == '__main__':
    args = parser.parse_args() 
    build_index(128, 30)                                               ## Vector_length and number of trees 
    print('build index is okay...')

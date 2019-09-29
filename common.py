# Load in the relevant packages
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session  

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
import gc
from keras import backend as K 

from sklearn.datasets import load_files   
from sklearn.model_selection import train_test_split
from keras.utils import np_utils 
import numpy as np
from glob import glob
from PIL import ImageFile, Image  
import re
from collections import Counter
import pandas as pd
from tqdm import tqdm
import itertools
import os
import datetime

from tensorflow.keras.preprocessing import image  
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential, load_model, model_from_json
from tensorflow.keras.callbacks import ModelCheckpoint 

if not os.path.exists('fruits/internet_resized'):
    os.makedirs('fruits/internet_resized')

# Define functions
def load_dataset(path, dictionary):
    print(path)
    start=datetime.datetime.now()
    data = load_files(path, load_content=False)
    dataload=datetime.datetime.now()
    print(str(dataload-start))
    files = np.array(data['filenames'])
    filepull=datetime.datetime.now()
    print(str(filepull-dataload))
    targets=([dictionary[re.sub(r'\d+', '',x.split('\\')[1]).strip()] for x in files])
    targetDetermine=datetime.datetime.now()
    print(str(targetDetermine-filepull))
    count=Counter(targets)
    counterCreate=datetime.datetime.now()
    print(str(counterCreate-targetDetermine))
    targets = np_utils.to_categorical(np.array(targets), len(dictionary))
    targetCategory=datetime.datetime.now()
    print(str(targetCategory-counterCreate))
    print("Total time : "+str(targetCategory-start))
    return files, targets, count

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(100, 100))
    # convert PIL.Image.Image type to 3D tensor with shape (100, 100, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 100, 100, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def resizeImages(path):
	# function used to resive any image to 100px by 100px for testing of this model
    images=glob(path+'/*')
    basewidth = 100
    new_path=[]
    targets=[]
    for ima in images:
        (image_dir,image_name)=ima.split('\\')
        targets.append(" ".join(image_name.split("_")[:-1]))
        img=Image.open(ima)
        wpercent = (basewidth/float(img.size[0]))
        hsize = int((float(img.size[1])*float(wpercent)))
        img = img.resize((basewidth,hsize), Image.ANTIALIAS)
        img.save(image_dir+'_resized\\'+image_name)
        new_path.append(image_dir+'_resized\\'+image_name)
    return new_path,targets

def getDataframes(tensors,targets,name):
    intensity=pd.DataFrame(list(zip([invFruitDict[np.argmax(x)] for x in targets],[(np.mean(x, axis=2)).mean() for x in tensors],[(np.std(x, axis=2)).std() for x in tensors])), columns= ['Fruit','Average Intensity','STD Intensity'])
    print('The average intensity of the '+name+' images is: %.4f with an standard deviation of %.4f.' % (intensity['Average Intensity'].mean(axis=0),intensity['STD Intensity'].mean(axis=0)))
    return intensity
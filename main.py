# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 23:37:18 2019

@author: Matthew
"""

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session  

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


from sklearn.datasets import load_files   
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import numpy as np
from glob import glob
from PIL import ImageFile  
import re
from collections import Counter
import pandas as pd
from tqdm import tqdm
from statistics import mean 
from PIL import Image

from tensorflow.keras.preprocessing import image  
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import ModelCheckpoint  

import datetime

"""
def load_dataset(path):
    data = load_files(path)
    files = np.array(data['filenames'])
    targets = np_utils.to_categorical(np.array(data['target']), 103)
    return files, targets
"""
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
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def resizeImages(path):
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

print("Getting files")
fruits = list(set([ re.sub(r'\d+', '', item[16:-1]).strip() for item in sorted(glob("fruits/Training/*/"))]))
fruitDict=dict(zip(fruits,list(range(0,len(fruits)+0))))
invFruitDict=dict(zip(list(range(0,len(fruits)+0)),fruits))

training_files, training_targets, training_counts = load_dataset('fruits/Training',fruitDict)
#training_files, training_targets = load_dataset('fruits/Training')
train_files, valid_files, train_targets, valid_targets= train_test_split( training_files, training_targets, test_size=0.15, random_state=1 )
test_files, test_targets, test_counts = load_dataset('fruits/Test',fruitDict)
#test_files, test_targets = load_dataset('fruits/Test')

training = {'Target':[invFruitDict[x] for x in training_counts.keys()],'Count':list(training_counts.values())}
training_df = pd.DataFrame(training)

testing = {'Target':[invFruitDict[x] for x in test_counts.keys()],'Count':list(test_counts.values())}
testing_df = pd.DataFrame(testing)

ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
print("Preprocessing Data")
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255

model = Sequential()

### TODO: Define your architecture.

model.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu', 
                        input_shape=(100, 100, 3)))
model.add(MaxPooling2D(pool_size=3))
model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=3))
model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=3))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(87, activation='softmax'))

model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 5

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', verbose=1, save_best_only=True)
print("Running model")
model.fit(train_tensors, train_targets, validation_data=(valid_tensors, valid_targets), epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)

model.load_weights('saved_models/weights.best.from_scratch.hdf5')

#model.save('saved_models/benchmark.h5')

model = load_model('saved_models/three_layer.h5')

# get index of predicted dog breed for each image in test set
print("Getting predicitions for testing set")
predictionArray=[model.predict(np.expand_dims(tensor, axis=0)) for tensor in tqdm(test_tensors)]

fruit_predictions = [np.argmax(prediction) for prediction in predictionArray]

#Determine the image with the most 0.01 or higher predicitions
pred = [list(prediction[np.where(prediction>0.01)])for prediction in predictionArray]
locMostPred=np.argmax([len(x) for x in pred])
imageMostPred=test_files[locMostPred]
lists=pred[locMostPred]
fruit=list((np.asarray(fruits))[np.where(predictionArray[6156]>0.01)[1]])

most=pd.DataFrame(list(zip(fruit,lists)), columns=['fruit','percentage'])

print("The image with the most predicitions above 1% is {}".format(imageMostPred))
print(most.sort_values(['percentage']))

averagePredPerImage=mean([len(x) for x in pred])

print("The average number of predicitions per image is {0:.4f}".format(averagePredPerImage))

#Determine the lowest score
lowestScore=min(([np.amax(prediction) for prediction in predictionArray]))
imageWithLowestScore=test_files[np.argmin(([np.amax(prediction) for prediction in predictionArray]))]
print('The Image with the lowest score of {0:.4f} is {1}'.format(lowestScore,imageWithLowestScore))

# report test accuracy
test_accuracy = 100*np.sum(np.array(fruit_predictions)==np.argmax(test_targets, axis=1))/len(fruit_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)


(internet_paths,targets)=resizeImages('fruits/internet')
internet_tensors= paths_to_tensor(internet_paths).astype('float32')/255
internet_array=[model.predict(np.expand_dims(tensor, axis=0)) for tensor in internet_tensors]  
internert_fruit_prediction_percent=[np.amax(prediction) for prediction in internet_array]       
internet_fruit_predictions = [np.argmax(prediction) for prediction in internet_array]   
internet_fruit_target_prediction=[prediction[0][fruitDict[targets[i]]] for i,prediction in enumerate(internet_array)]                 
internet_test_accuracy=100*np.sum((np.array(targets))==(np.asarray(fruits)[internet_fruit_predictions]))/len(targets)
print('Internet Image accuracy: %.4f%%' % internet_test_accuracy)
report=pd.DataFrame({'path':internet_paths, 'target':targets, 'prediction':(np.asarray(fruits)[internet_fruit_predictions]), 'predictionPercent':internert_fruit_prediction_percent, 'targetPredictionPercent':internet_fruit_target_prediction})

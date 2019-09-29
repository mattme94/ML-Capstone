# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 23:37:18 2019

@author: Matthew
"""

# import the self-made common module that contains all the functions and other packages required
from common import *

# Get the files and create two dictionaries to convert the fruit name to the target and the target back to the fruit name
print("Getting files")
fruits = list(set([ re.sub(r'\d+', '', item[16:-1]).strip() for item in sorted(glob("fruits/Training/*/"))]))
fruitDict=dict(zip(fruits,list(range(0,len(fruits)+0))))
invFruitDict=dict(zip(list(range(0,len(fruits)+0)),fruits))

# Load the datasets in 
training_files, training_targets, training_counts = load_dataset('fruits/Training',fruitDict)
train_files, valid_files, train_targets, valid_targets= train_test_split( training_files, training_targets, test_size=0.15, random_state=1 )
test_files, test_targets, test_counts = load_dataset('fruits/Test',fruitDict)

ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
print("Preprocessing Data")
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255

# Process the internet sourced images
(internet_paths,targets)=resizeImages('fruits/internet')
internet_tensors= paths_to_tensor(internet_paths).astype('float32')/255

# Create the parameters to iterate on                              
dropout1=[0.2+0.1* x for x in list(range(4))]
dropout2=[0.2+0.1* x for x in list(range(4))]                            
optimizer = ['Adagrad','SGD', 'RMSprop', 'Adam']     

# Create a dataframe
grid_search = [dropout1, dropout2,optimizer]
grid_search_data = list(itertools.product(*grid_search))
grid_search_idx = [i for i in range(1, len(grid_search_data)+1)]

grid_search_df = pd.DataFrame(grid_search_data, index=grid_search_idx, columns=['dropout1','dropout2','optimizer'])                   

grid_search_df['trainAccuracy']=np.nan
grid_search_df['testAccuracy']=np.nan
grid_search_df['learningTime']=np.nan
grid_search_df['internetAccuracy']=np.nan

# if variable refinement does not exist, create it
if not os.path.exists('variable_refinement'):
    os.makedirs('variable_refinement')

# Iterate through the process of create the model and then testing it           
for index,row in grid_search_df.iterrows():
    print(row)
    
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu',input_shape=(100, 100, 3)))
    model.add(MaxPooling2D(pool_size=3))
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(row['dropout1']))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(row['dropout2']))
    model.add(Dense(len(fruits), activation='softmax'))
      
    model.compile(optimizer=row['optimizer'], loss='categorical_crossentropy', metrics=['accuracy'])
    
    epochs = 5
    
    checkpointer = ModelCheckpoint(filepath='variable_refinement/weights.best.from_scratch.hdf5', verbose=0, save_best_only=True)
    print("Running model")
    start=datetime.datetime.now()
    history=model.fit(train_tensors, train_targets, validation_data=(valid_tensors, valid_targets), epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)
    end=datetime.datetime.now() 
    print(str(end-start))
    grid_search_df.at[index,'trainAccuracy']=history.history['acc'][-1]
    grid_search_df.at[index,'learningTime']=end-start
    model.load_weights('variable_refinement/weights.best.from_scratch.hdf5')
    
    model.save('variable_refinement/'+(str (index))+'.h5')
    
     # Get test set accuracy
    print("Getting predicitions for testing set")
    predictionArray=[model.predict(np.expand_dims(tensor, axis=0)) for tensor in test_tensors]
    fruit_predictions = [np.argmax(prediction) for prediction in predictionArray]
    test_accuracy = 100*np.sum(np.array(fruit_predictions)==np.argmax(test_targets, axis=1))/len(fruit_predictions)
    print('Test accuracy: %.4f%%' % test_accuracy)
    grid_search_df.at[index,'testAccuracy']=test_accuracy
    
    # Compute the accuracy on the internet images 
    internet_array=[model.predict(np.expand_dims(tensor, axis=0)) for tensor in internet_tensors]  
    internert_fruit_prediction_percent=[np.amax(prediction) for prediction in internet_array]       
    internet_fruit_predictions = [np.argmax(prediction) for prediction in internet_array]   
    internet_fruit_target_prediction=[prediction[0][fruitDict[targets[i]]] for i,prediction in enumerate(internet_array)]                 
    internet_test_accuracy=100*np.sum((np.array(targets))==(np.asarray(fruits)[internet_fruit_predictions]))/len(targets)
    print('Internet Image accuracy: %.4f%%' % internet_test_accuracy)
    report=pd.DataFrame({'path':internet_paths, 'target':targets, 'prediction':(np.asarray(fruits)[internet_fruit_predictions]), 'predictionPercent':internert_fruit_prediction_percent, 'targetPredictionPercent':internet_fruit_target_prediction})
    grid_search_df.at[index,'internetAccuracy']=internet_test_accuracy
    report.to_csv('variable_refinement/'+(str (index))+'.csv')
    grid_search_df.to_csv('gridSearch.csv')
    K.clear_session()
    gc.collect()
    del model
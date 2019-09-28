# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 23:37:18 2019

@author: Matthew
"""

# import the self-made common module that contains all the functions and other packages required
import common

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

# Start the iteration   
# Find all the json files (saved model)                       
models=glob('model_refinement'+'/*.json')

models_idx = [i for i in range(1, len(models)+1)]
# Create a dataframe 
model_refinement_df = pd.DataFrame(models, index=models_idx, columns=['fileName'])                   

model_refinement_df['trainAccuracy']=np.nan
model_refinement_df['testAccuracy']=np.nan
model_refinement_df['learningTime']=np.nan
model_refinement_df['internetAccuracy']=np.nan
           
model_refinement_df.to_csv('model_refinement_df.csv',index_label='index')

# Iterate through that dataframe
for index,row in model_refinement_df.iterrows():
	# Reload the dataframe
	model_refinement_df=pd.read_csv('model_refinement_df.csv',index_col='index')
	# If the dataframe already has results for this row, do not need to test again
    if not np.isnan(row['trainAccuracy']):
        print('Opening '+row['fileName'])   
    else:
        print('Opening '+row['fileName'])  
        # Open the model and load it in
        with open(row['fileName'],'r') as file:
            model = model_from_json(file.readline()) 
        # Compile the model	
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        epochs = 5
        # Use a checkpointer to save base version of model
        checkpointer = ModelCheckpoint(filepath='model_refinement/weights.best.from_scratch.hdf5', verbose=1, save_best_only=True)
        print("Running model")
        start=datetime.datetime.now()
        history=model.fit(train_tensors, train_targets, validation_data=(valid_tensors, valid_targets), epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)
        end=datetime.datetime.now() 
        print(str(end-start))
        # Add the training accuracy and the time taken to the dataframe 
        model_refinement_df.at[index,'trainAccuracy']=history.history['acc'][-1]
        model_refinement_df.at[index,'learningTime']=end-start

        # Load the best model and save it down 
        model.load_weights('model_refinement/weights.best.from_scratch.hdf5')
        model.save('model_refinement/'+(str (index))+'.h5')
        
        # Get test set accuracy
        print("Getting predicitions for testing set")
        predictionArray=[model.predict(np.expand_dims(tensor, axis=0)) for tensor in test_tensors]
        fruit_predictions = [np.argmax(prediction) for prediction in predictionArray]
        test_accuracy = 100*np.sum(np.array(fruit_predictions)==np.argmax(test_targets, axis=1))/len(fruit_predictions)
        print('Test accuracy: %.4f%%' % test_accuracy)
        # Add test set accuracy to report
        model_refinement_df.at[index,'testAccuracy']=test_accuracy
        
        # Compute the accuracy on the internet images
        internet_array=[model.predict(np.expand_dims(tensor, axis=0)) for tensor in internet_tensors]  
        internert_fruit_prediction_percent=[np.amax(prediction) for prediction in internet_array]       
        internet_fruit_predictions = [np.argmax(prediction) for prediction in internet_array]   
        internet_fruit_target_prediction=[prediction[0][fruitDict[targets[i]]] for i,prediction in enumerate(internet_array)]                 
        internet_test_accuracy=100*np.sum((np.array(targets))==(np.asarray(fruits)[internet_fruit_predictions]))/len(targets)
        print('Internet Image accuracy: %.4f%%' % internet_test_accuracy)

        # Create a report of the internet predicitions 
        report=pd.DataFrame({'path':internet_paths, 'target':targets, 'prediction':(np.asarray(fruits)[internet_fruit_predictions]), 'predictionPercent':internert_fruit_prediction_percent, 'targetPredictionPercent':internet_fruit_target_prediction})
        model_refinement_df.at[index,'internetAccuracy']=internet_test_accuracy
        report.to_csv('model_refinement/'+(str (index))+'.csv')
        # Save the dataframe down
        model_refinement_df.to_csv('model_refinement_df.csv',index_label='index')
        # Clear the session and delete model to prevent GPU memory from being used up
        K.clear_session()
        gc.collect()
        del model
        
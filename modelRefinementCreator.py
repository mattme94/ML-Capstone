# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 23:37:18 2019

@author: Matthew
"""

import common

#Define the variables to be used
kernel_sizes=[[5,5,5,5,5],[7,5,5,5,5],[7,7,5,5,5],[7,7,7,7,5],[7,7,7,7,7],[3,5,5,5,5],[3,3,5,5,5],[3,3,3,3,5],[3,3,3,3,3]]
pool_sizes=[[2,2,2,2,2],[3,2,2,2,2],[3,3,2,2,2],[3,3,3,2,2],[3,3,3,3,3]]
layers=[0,1,2]
layersDict={0:'3Conv2DLayers',1:'Base',2:'5Conv2DLayers'}
denses1=[1024,2048]
denses2=[256,512]

# if variable refinement does not exist, create it
if not os.path.exists('model_refinement'):
    os.makedirs('model_refinement')

# Iterate through all the variables to produce the models
for layer in layers:
    for kernel,kernel_size in enumerate(kernel_sizes):
        # Check that if layer==2 (aka layer 5) is being added so that the pool size is not increased creating an invalid model
        if layer==2:
            Pool_sizes=[pool_sizes[0]]
        else:
            Pool_sizes=pool_sizes
        for pool,pool_size in enumerate(Pool_sizes):
            for dense1 in denses1:
                for dense2 in denses2:
                    model = Sequential()
                    model.add(Conv2D(filters=16, kernel_size=kernel_size[0], padding='same', activation='relu',input_shape=(100, 100, 3)))
                    model.add(MaxPooling2D(pool_size=pool_size[0]))
                    model.add(Conv2D(filters=32, kernel_size=kernel_size[1], padding='same', activation='relu'))
                    model.add(MaxPooling2D(pool_size=pool_size[1]))
                    model.add(Conv2D(filters=64, kernel_size=kernel_size[2], padding='same', activation='relu'))
                    model.add(MaxPooling2D(pool_size=pool_size[2]))
                    # Check the layer number to see what layers should be added
                    if layer>0:
                        model.add(Conv2D(filters=128, kernel_size=kernel_size[3], padding='same', activation='relu'))
                        model.add(MaxPooling2D(pool_size=pool_size[3]))
                        if layer>1:
                            model.add(Conv2D(filters=256, kernel_size=kernel_size[4], padding='same', activation='relu'))
                            model.add(MaxPooling2D(pool_size=pool_size[4]))
                    model.add(Flatten())
                    model.add(Dense(dense1, activation='relu'))
                    model.add(Dropout(0.2))
                    model.add(Dense(dense2, activation='relu'))
                    model.add(Dropout(0.2))
                    model.add(Dense(87, activation='softmax'))
                    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                    json_string = model.to_json()
                    with open('model_refinement/model_'+layersDict[layer]+';dense1_'+str(dense1)+';dense2_'+str(dense2)+';maxpooling_pool_size_'+str(pool_sizes[1][1])+'_to_'+str(pool_sizes[1][0])+'_layer'+str(pool)+';conv2d_kernel_size_'+str(kernel_sizes[1][1])+'_to_'+str(kernel_sizes[1][0])+'_layer'+str(kernel)+'.json', 'w') as file:
                        file.write(json_string)
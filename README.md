# ML-Capstone

This is the git repo for my machine learning nanodegree capstone project. I examined the use of CNNs to indentify fruits from still images. This was built and ran on python 3.7.3. 

There are five python scripts in the repository:

* common.py contains the common functions as well as all of the importing as packages

* modeRefinementCreator.py creates the different model architectures as discussed in the report
 
* modelRefinement.py tests the model created by the above script

* parameterTuning.py tests the best model from the modelRefinement script (chosen manually) with variations on dropout percentages and optimizer

* final.py contains the final representation of the model

## Required Packages

* keras 2.2.4 (I used the GPU version) 
* pandas 0.24.2
* numpy 1.16.4 
* tqdm 4.32.1
* sklearn 0.21.2
* pillow 6.1.0 
* tensorflow 1.14.0 

## Dataset

The dataset can be downloaded [here](kaggle.com/moltean/fruits/ "Dataset").

The data should be placed in the *fruits* folder so that the *Test* and *Training* folders of the dataset are on the same level as *internet*.
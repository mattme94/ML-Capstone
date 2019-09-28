# ML-Capstone

There are five python scripts in the repository:

* common.py contains the common functions as well as all of the importing as packages

* modeRefinementCreator.py creates the different model architectures as discussed in the report
 
* modelRefinement.py tests the model created by the above script

* parameterTuning.py tests the best model from the modelRefinement script (chosen manually) with variations on dropout percentages and optimizer

* final.py contains the final representation of the model

## Required Packages

* keras (i used gpu) 
* pandas
* glob
* numpy
* tqdm
* sklearn
* PIL
* tensorflow

## Dataset

The dataset can be downloaded [here](http://https://www.kaggle.com/moltean/fruits/ "Dataset").

The data should be placed in the *fruits* folder so that the *Test* and *Training* folders of the dataset are on the same level as *internet*.
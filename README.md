# Bachelor thesis source code
**Title**: PREDICTIVE MODELING FOR ENHANCED PERFORMANCE: EXPLORING MACHINE LEARNING TECHNIQUES TO OPTIMISE PEROVSKITE SOLAR CELL DESIGN

## Description
### Source code
You can find the code to train the models, and produce the figures in *ThesisCode.m* and *ThesisCode.mlx*. These are equivalent with the difference 
of *.mlx* file offering a better visal rendering.

There is also standalone code to train a neural network to predict power conversion efficiency (PCE) available in *standalone.m* file.

### Importing models
The Neural Network regression models trained on the 100k dataset can be found in the *NN* folder. 
To import the models into your code use the following command: ```load('NN/Y_*_NN.mat')```, 
where * is the number of the model you want to import.

### Dataset
The 10k dataset can be found in the *Data_10_k* folder.
*LHS_parameters_m.txt* is the input data, representing the parameters of the perovskite solar cells.
*CellPerformance.txt* is the output data containing the solar cell key metric values for each solar cell design.

Further explanations on data and parameters are provided in the *Explanations on data.pdf* file.
The 100k dataset was too large to be uploaded, hence it is not presented here.
The same goes for some other objects like, the SHAP object containg the Shapley values for all predictors in the NN model to predict
solar cell power conversion efficiency.

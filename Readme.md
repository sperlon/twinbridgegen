# Twinbridgegen
This repo contains upgraded version of app developed by Karel Mařík. 

I modified the modes
of the app, so that it now runs in three modes:
- ### isTraining
  - In this mode the model is trained, and evaluated on the testing data. The data for training
  are loaded from .xlsx file specified by trainInPN argument.
  - The trained model is stored in a path specified by modelPN argument.
  - You can choose from three type of models, see Model section bellow,
- ### isPrediction
  - During this mode, the trained model is loaded from path specified by modelPN
  argument, and it does the prediction on the data loaded form .xlsx file specifed
  by predInPN parameter.
- ### isInverseAnalysis
  - In this mode, the app will load the saved model and preform inverse analysis. For
  this mode you must provide model outputs in the form of .xlsx file with path specified
  by invAnInPN and the algorithm will iteratively search for corresponding input using SGD.
  You can adjust this method with following parameters:
    - inverseLearningRate: Controls the step size of SGD.
  
    - inverseNIter: Number of independent optimization runs for better exploration.
  
    - inverseBestOnly: If true, only the best input found across runs is returned. If false, all the iterations are return in final .csv files
  
    - inverseLowerLim / inverseUpperLim: Define the allowed range for each input dimension. 
  
    - inverseToler: The convergence criterion; optimization stops when Loss value (Residuum) is below this value. The Loss is calculated as L2 norm between models produced output and desired output. 
  
    - inverseMaxIter: Maximum iterations for each SGD run.
  
    - inversePrintFreq: How often progress updates are displayed.
  - The structure of the final .csv is as follows: Output ID, Iteration ID, Loss value (Residuum), Model input, Model output.
## Other parameters:
- ### Running modes
  - isTraining: If specified, the model is trained.
  - isPrediction: If specified, data from path predictionDataFN are loaded, and trained model specified by the modelPN path is used for the prediction.
  - isInverseAnalysis: If specified, the inverse analaysis is preformed, see the the upper text.
- ### Model specification
  -  modelType: Specifies the type of the model. You must pass one of following strings 'denseModel', 'dividedDenseModel', or 'multiChannelModel'. The dense model is simple consisting only of fully conected layers (dense layers). This is the same model type that was used in the original app. Divided dense model is also consisting only of fully connected layers, but for each member in the output, seperate sub network is used. Every subnetwork has same shape. And for multichannel model you can choose from either densely connected layers or lstm layers. Beside that you can specify multiple channels that are in the end added to one output. For each channel you can specify different inputs, see the instructions bellow.
  -  #### Dense model
    -  denseLayers: A string containing list of layers with integer specifying number of neurons in each layer. For example '128, 128, 128' would result in model with three hidden layers with 128 neurons each. The size of the input and the output layer is determined automatically, so you should only specify the sizes of the hidden layers.
  - #### Divided dense model
    - dividedLayers: A string containing list of layers with integer specifying number of neurons in each layer. Works the same way as denseLayers, but this time specified architecture is apllied to every member in the output separetely. For example if the model output is 2 dimensional (i.e. vector of 2 - it could be 2 strains etc. ) and you specify '128, 128, 128' again, different submodel with three hidden layers with 128 neurons each is applied to each member of the output (each strain).
  - #### Multichannel model
    - multiLayers: A string containing the python list with definition of the model. The definition use the following logic. The main list is containg two levels of nested lists. The first nested list is defining one channel (subomdel) itself and it contains the second nested list which are specifying the layers within the submodel. The layer list can be of two types. 1)['LSTM', int - specifying the number of neurons], ['Dense', int - specifying the number of neurons, string - specyfing the name of the activation function (You can use any activation function from Keras, see official Keras documentation for available functions)]. In the end, the output from each channel is added together, so remeber the last layer in each channel must contain the same number of neurons! For example "[[['LSTM', 16], ['Dense', 16, 'relu']], [['Dense', 16, 'relu'], ['Dense', 16, 'relu']]]" would result with model with two channels. First chanel would contain LSTM layer with 16 neurons, followed by dense layer also with 16 neurons and relu activation. Second channel would contain two dense layers, again with 16 neurons each and relu activation function. The output from this channel would be added together to one vector with length of 16 and passed to the final output layer.
    - multiInput: A string containing python list with specification of the input to each channel of the model. Each input is specified again with the list containing either the names of the columns in .xlsx file used for training, or the index of this column starting with zero. For example '[[0, 1], [2, 3]]' would result with first two column passed to the first channel and the third and fourth column passed to the second chanel. Assuming that the name of this four columns are 'strain1', ..., 'strain4' you could also get the same result with '[["strain1", "strain2"], ["strain3", "strain4"]]'.
- ### Training of the model
  - testSize: The number between 0 and 1 specifying the precentage of data that will be used for testing
  - epochN: The number of epochs for which the model will be trained
  - learningRate: The value for the learining rate for the training

For all of the parameters, you can check twinBridgeGen.py
  

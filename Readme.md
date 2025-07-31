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
  - In this mode, the app will load the save model and preform inverse analysis. For
  this mode you must provide model outputs in the form of .xlsx file with path specified
  by invAnInPN and the algorithm will iteratively search for corresponding input using SGD.
  You can adjust this method with following parameters:
    - inverseLearningRate: Controls the step size of SGD.
  
    - inverseNIter: Number of independent optimization runs for better exploration.
  
    - inverseBestOnly: If true, only the best input found across runs is returned.
  
    - inverseLowerLim / inverseUpperLim: Define the allowed range for each input dimension.
  
    - inverseToler: The convergence criterion; optimization stops when improvement is below this value.
  
    - inverseMaxIter: Maximum iterations for each SGD run.
  
    - inversePrintFreq: How often progress updates are displayed.

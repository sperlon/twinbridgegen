# Author:  Radek Marik, radek.marik@fel.cvut.cz
# Created: Dec 21, 2023
# Purpose: parameter learning

# Copyright (c) 2023, Radek Marik, FEE CTU, Prague
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of FEE CTU nor the
#       names of contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import print_function
import os.path, random, copy, pickle, time, csv
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Reshape
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.utils import plot_model
from LibPy.ParallelWorkers import WorkerManager_C
from LibPy import GraphicWorkers
Set_C = set

# =======================================================================
# ===================== Support Structures ==============================
# =======================================================================

# =======================================================================
# ===================== Support Functions ===============================
# =======================================================================

def PlotSpec(setupOptions):
  #========================
  prop_cycle = plt.rcParams['axes.prop_cycle']
  colors     = prop_cycle.by_key()['color']

  return {
    'outPN':   pathlib.Path(setupOptions.outPN, setupOptions.datasetDir, setupOptions.expTag),
    'cMap':    'jet',
    'colors':  colors,
  }
    
def DenseModel(setupOptions, runCfg, xN, yN, layers, xTrain, yTrain):
  #==================================================================
  epochN          = setupOptions.epochN
  verbose         = setupOptions.verbose
  learningRate    = runCfg['training']['learningRate']
  loss            = runCfg['training']['loss']
  metrics         = runCfg['training']['metrics'].split(',')
  validationSplit = runCfg['training']['validationSplit']

  # ANN model
  inp = Input((xN,))
  lastLayer = inp
  for layerN in layers:
    lastLayer = Dense(layerN, activation = 'relu')(lastLayer)
  out = Dense(yN)(lastLayer)
  annModel = Model(inputs=inp, outputs=out)
  annModel.compile(optimizer=Adam(learning_rate=learningRate), loss=loss, metrics=metrics)
  start = time.time()
  history = annModel.fit(xTrain, yTrain, epochs=epochN, validation_split = validationSplit, verbose = verbose)
  end = time.time()
  if verbose > 2: 
    print('learning time:', end - start)
    annModel.summary()
  return annModel, history


def WriteToCsv(data, path, header=None):
  with open(path, 'wt') as csvF:
    parWriter = csv.writer(csvF, delimiter=',')
    if header:
      parWriter.writerow(header)
    for yRow in data:
      parWriter.writerow(yRow)

def CheckLSTMLayer(layer):
  n_args = len(layer)
  if n_args != 2:
    raise Exception(f"LSTM layer is incorrectly specified. The list must have exactly 2 arguments(LayerName:string, nWeights:int), but has {n_args}({layer}).")
  elif not isinstance(layer[1], int) and layer[1] > 0:
    raise Exception(f"Number of neurons must be non-zero non-negative integer, but you specified: {layer[1]}")


def CheckDenseLayer(layer):
  n_args = len(layer)
  if n_args == 2:
    layer.append(None)
    n_args += 1

  if n_args != 3:
    raise Exception(f"""Dense layer is incorrectly specified. The list should have 3 arguments(LayerName:string, nWeights:int, ActivationFunc:str), but has {n_args}({layer}).
     It is also possible to pass only 2 arguments(LayerName:string, nWeights:int), but in such case activation func would be None.""")

  elif not isinstance(layer[1], int) and layer[1] > 0:
    raise Exception(f"Number of neurons must be non-zero non-negative integer, but you specified: {layer[1]}")

  elif not isinstance(layer[2], str):
    if layer[2] is None:
        pass
    else:
      raise Exception(f"Activation function must be specified as string or none, but is {type(layer[0])}({layer[2]}) instead.")

def SubModel(layers, input):
  x = input
  n_layers = len(layers)

  first = True
  for i in range(n_layers):
    layer = layers[i]
    if layer[0].lower() == "lstm":
      CheckLSTMLayer(layer)
      # check whether the lstm layer is the last lstm in a row. If not return, full sequence
      if i == n_layers-1:
        ret_seq = False
      elif layers[i+1][0].lower() != "lstm":
        ret_seq = False
      else:
        ret_seq = True
      if first: # If LSTM is the first layer, it expects two-dimensional input. Hence we must manually reshape it
        x = tf.keras.layers.Reshape((x.shape[-1], 1))(x)
        first = False
      x = tf.keras.layers.LSTM(layer[1], return_sequences=ret_seq)(x)

    elif layer[0].lower() == "dense":
      CheckDenseLayer(layer)
      if first:
          first = False
      x = tf.keras.layers.Dense(layer[1], activation=layer[2])(x)

    else:
      raise Exception(f"Can not recognize layer: '{layer[0]}'.")

  return x


def MultiChannelModel(layers, input_dims, out_dim):
  inputs = [tf.keras.layers.Input(shape=[inp_dim]) for inp_dim in input_dims]
  sub_outputs = []

  for i in range(len(layers)):
    x = SubModel(layers[i], inputs[i])
    sub_outputs.append(x)

  x = tf.keras.layers.Add()(sub_outputs)
  out = tf.keras.layers.Dense(out_dim)(x)

  model = tf.keras.Model(inputs=inputs, outputs=out)
  return model


# =======================================================================
# ===================== Classes =========================================
# =======================================================================

class RangeScaler_C(object):
  #-------------------------
  
  def __init__(self, isFlatten = False):
    #===================================
    self.minV = None 
    self.maxV = None 
    self.isFlatten = isFlatten

  def Save(self, outFN):
    #===================
    with open(outFN, 'wb') as outF:
      pickle.dump({'minV': self.minV, 'maxV': self.maxV, 'isFlatten': self.isFlatten}, outF)

  def Load(self, outFN):
    #===================
    with open(outFN, 'rb') as outF:
      myPars = pickle.load(outF)
    self.minV = myPars['minV']
    self.maxV = myPars['maxV']
    self.isFlatten = myPars['isFlatten']
            
  def fit(self, data):
    #=================
    if 1: print('RangeScaler_C.data', data.shape)
    if self.isFlatten:
      flattenData = data.flatten()
      self.minV = np.min(flattenData)
      self.maxV = np.max(flattenData)
    else:
      dimN = data.ndim
      if dimN == 3:
        self.minV = [np.min(data[:,:,i]) for i in range(data.shape[-1])]
        self.maxV = [np.max(data[:,:,i]) for i in range(data.shape[-1])]
      elif dimN == 2:
        self.minV = [np.min(data[:,i]) for i in range(data.shape[-1])]
        self.maxV = [np.max(data[:,i]) for i in range(data.shape[-1])]
      elif dimN == 1:
        self.minV = np.min(data)
        self.maxV = np.max(data)
      
    if 0: print('minMax', self.minV, self.maxV)
    
  def transform(self, data):
    #=======================
    if self.isFlatten:
      return (data-self.minV)/(self.maxV-self.minV)
    dimN = data.ndim
    if dimN == 3:
      return np.dstack([(data[:,:,i]-self.minV[i])/(self.maxV[i]-self.minV[i]) for i in range(data.shape[dimN-1])])
    elif dimN == 2:
      return np.dstack([(data[:,i]-self.minV[i])/(self.maxV[i]-self.minV[i]) for i in range(data.shape[dimN-1])])[0]
    elif dimN == 1:
      return (data-self.minV)/(self.maxV-self.minV)
  
  def fit_transform(self, data):
    #===========================
    self.fit(data)
    return self.transform(data)
  
  def inverse_transform(self, data):
    #===============================
    if self.isFlatten:
      return data * (self.maxV-self.minV) + self.minV
    dimN = data.ndim
    if dimN == 3:
      return np.dstack([(data[:,:,i] * (self.maxV[i]-self.minV[i]) + self.minV[i]) for i in range(data.shape[dimN-1])])
    elif dimN == 2:
      return np.dstack([(data[:,i] * (self.maxV[i]-self.minV[i]) + self.minV[i]) for i in range(data.shape[dimN-1])])[0]
    elif dimN == 1:
      return data * (self.maxV-self.minV) + self.minV


class DividedModel(tf.keras.Model):
    """
    Class for creation of a model where each member in output vector is predicted with different subnetwork.
    It turned out that for prediction of the strain from material parameters, it works better to have one smaller subnetwork
    for each strain than have one big network that the whole strain vector at once.

    You just have to specify layers in layers argument that each subnetwork will have (for example '[64, 64, 64]' would create
    subnetwork with 3 hidden layers with 64 neurons each), dimension of input vector(number of material params), and dimension
    of output vector(number of strains to predict).

    I designed this architecture specifically for the prediction of strain from material parameters, but if it proves advantageous,
    it can be used anywhere.

    Further there are methods for searching a corresponding input to given output (I want to find material parameters for
    specified strains). The newton´s method, the Gauss-newton´s method and SGD - stochastic gradient descent. However, for
    such a purpose I recommend to use only SGD, since the first two are unstable and most the time unable to converge
     """

    def __init__(self, layers, inp_dim, out_dim, act="relu", **kwargs):
      super().__init__(**kwargs)
      self.inp_dim = inp_dim
      self.out_dim = out_dim
      self.sub_models = []
      # creation of submodels for each output parameter
      for o in range(out_dim):
        sub_model = tf.keras.Sequential([tf.keras.layers.Dense(layers[0], activation=act, input_shape=[inp_dim])])
        for l in range(1, len(layers)):
          sub_model.add(tf.keras.layers.Dense(layers[l], activation=act))
        sub_model.add(tf.keras.layers.Dense(1))
        self.sub_models.append(sub_model)

    # keras method that needs to be defined. It specifies how output is calculated
    def call(self, inputs):
      part_outputs = []  # store each member of the output vector (strain) in a list
      for o in range(self.out_dim):
        part_outputs.append(
          self.sub_models[o](inputs))  # make prediction of output member (strain) by subnetwork and store it
      output = tf.keras.layers.Concatenate(axis=1)(part_outputs)  # concatenate the output to final output vector
      return output

    # SGD - stochastic gradient descent
    # ------------------------------------------------------------------------------------------------------------------
    # Getting the gradient with respect to input
    @tf.function  # this is a decorator that specifies for tensorflow to convert this method into the computational graph. As a result the computation is significantly faster and can run on GPU
    def get_grad_output_input(self, output, input):
      with tf.GradientTape() as tape:
        pred = self.call(input)
        l = tf.reduce_sum(tf.square(output - pred))

      grad = tape.gradient(l, input)
      return grad, l

    # This method search for optimal input for given output using gradient descent.
    # You can specify lower and upper limit for each parameter, tolerance (L2 norm between model prediction and searched output),
    # number of iterations and print_freq
    def find_input_SGD_based(self, output, input, optimizer, lower_limit=None, upper_limit=None, tolerance=1e-3,
                             max_iter=500, print_freq=100):
      out_ = tf.cast(output, tf.float32)
      inp_ = tf.cast(input, tf.float32)
      inp0 = tf.Variable(inp_, trainable=True)
      result = inp0.value()

      for i in range(max_iter):
        grad, l = self.get_grad_output_input(out_, inp0)
        optimizer.apply_gradients([(grad, inp0)])

        # check whether the parameters don´t exceed the limits
        if lower_limit is not None:
          if tf.reduce_any(inp0 < lower_limit):
            print("Lower limit broken")
            break

        elif upper_limit is not None:
          if tf.reduce_any(inp0 > upper_limit):
            print("Upper limit broken")
            break

        result = inp0.value()

        # print the loss value after the specified frequency, last iteration or when tolerance is achieved
        if i == 0 or i == max_iter - 1:
          print(f"Iteration {i}: loss = {l.numpy()}")
        elif (i + 1) % print_freq == 0:
          print(f"Iteration {i}: loss = {l.numpy()}")
        if l <= tolerance:
          print(f"Iteration {i}: loss = {l.numpy()}")
          break
      return result


# =======================================================================
# ===================== Main Functions ==================================
# =======================================================================

def DataPreparation(setupOptions, runCfg, dataPool):
  #=================================================
  testSize          = setupOptions.testSize
  randomSeed        = runCfg['training']['randomSeed']
  trainingResultsPN =  setupOptions.trainResultsPN
  modelPN           = setupOptions.modelPN
  modelType = setupOptions.modelType


  # Create input/output data
  inSheetName = runCfg['trainingDataset']['inSheet']
  if setupOptions.inPars == '#ALL#':
    inParNames = dataPool[inSheetName]['colNames'][1:] # skip the index
  else:
    inParNames  = setupOptions.inPars.split(',')
  inData = pd.DataFrame(dataPool[inSheetName]['data'],
                        index   = dataPool[inSheetName]['index'],
                        columns = dataPool[inSheetName]['colNames'])
  inData = inData[inParNames].to_numpy()
  outSheetName = runCfg['trainingDataset']['outSheet']
  if setupOptions.outPars == '#ALL#':
    outParNames = dataPool[outSheetName]['colNames'][1:]  # skip the index
  else:
    outParNames = setupOptions.outPars.split(',')
  outData = pd.DataFrame(dataPool[outSheetName]['data'],
                         index=dataPool[outSheetName]['index'],
                         columns=dataPool[outSheetName]['colNames'])
  outData = outData[outParNames].to_numpy()
  if setupOptions.verbose > 1: print('inData', inData.shape)
  if setupOptions.verbose > 1: print('outData', outData.shape)

  plotSpec = PlotSpec(setupOptions)
    
  # Normalization
  xScaler = RangeScaler_C()
  xData   = xScaler.fit_transform(inData)
  yScaler = RangeScaler_C()
  yData   = yScaler.fit_transform(outData)

  wManager      = WorkerManager_C()

  if setupOptions.verbose > 1:
    xTmp    = xScaler.inverse_transform(xData)
    fName   = f'Simulation_xData.{setupOptions.imgExt}'
    xPlotSpec = plotSpec.copy()
    xPlotSpec['outPN'] = trainingResultsPN
    xPlotSpec['yPars']   = inParNames
    xPlotSpec['yGroups'] = runCfg['training']['xPlotGroups']
    xPlotSpec['yLabels'] = runCfg['training']['xPlotLabels']
    wManager.Run(GraphicWorkers.SimulationsWorker, (xPlotSpec, fName, xTmp))

  # Train/test split
  instanceMap = np.arange(len(xData))
  xTrainMap, xTestMap, yTrainMap, yTestMap = train_test_split(
    instanceMap, instanceMap, test_size=testSize, random_state=randomSeed)
  xTrain = xData[xTrainMap]
  xTest  = xData[xTestMap]
  yTrain = yData[yTrainMap]
  yTest  = yData[yTestMap]
  if 0: print('xy', xData.shape, yData.shape, xTrain.shape, xTest.shape, yTrain.shape, yTest.shape)

  # Persist all data
  np.save(trainingResultsPN / f'xTrain.npy',    xTrain)
  np.save(trainingResultsPN / f'xTest.npy',     xTest)
  np.save(trainingResultsPN / f'yTrain.npy',    yTrain)
  np.save(trainingResultsPN / f'yTest.npy',     yTest)
  np.save(trainingResultsPN / f'xTrainMap.npy', xTrainMap)
  np.save(trainingResultsPN / f'xTestMap.npy',  xTestMap)
  np.save(trainingResultsPN / f'yTrainMap.npy', yTrainMap)
  np.save(trainingResultsPN / f'yTestMap.npy',  yTestMap)
  with open(modelPN / 'outParNames.pck', 'wb') as outF:
    pickle.dump(outParNames, outF)
  with open(modelPN / 'inParNames.pck', 'wb') as outF:
    pickle.dump(inParNames, outF)
  xScaler.Save(modelPN / f'xScaler.pck')
  yScaler.Save(modelPN / f'yScaler.pck')

  wManager.Join()

def ParameterModelTraining(setupOptions, runCfg):
  #==============================================
  trainingResultsPN = setupOptions.trainResultsPN
  modelPN = setupOptions.modelPN
  annDLayers = [int(s) for s in setupOptions.denseLayers.split(',')]
  xTrain = np.load(trainingResultsPN / f'xTrain.npy')
  yTrain = np.load(trainingResultsPN / f'yTrain.npy')
  xN = len(xTrain[0])
  yN = len(yTrain[0])
  annModel, history = DenseModel(setupOptions, runCfg, xN, yN, annDLayers, xTrain, yTrain)
  annModel.save(modelPN / f'annModel.keras')

  if setupOptions.verbose > 2:
    # Some graphical output
    plot_model(annModel, to_file= modelPN / f'annModel_Model.png',
               show_shapes=True, show_layer_names=True)
    print(f"Loss: {history.history['loss'][-1]}")
    print(f"validation loss: {history.history['val_loss'][-1]}")
    wManager = WorkerManager_C()

    plotDsc = {
    'outPN':          trainingResultsPN,
    'graphBaseName':  f'annModel_TrainHistory.png',
    'history':        history.history,
    }
    wManager.Run(GraphicWorkers.TrainingHistoryWorker, (plotDsc,))

    wManager.Join()

def ParameterModelEvaluation(setupOptions):
  # ===============================================
  trainingResultsPN = setupOptions.trainResultsPN
  modelPN = setupOptions.modelPN


  with open(modelPN / 'outParNames.pck', 'rb') as inF:
    outParNames = pickle.load(inF)

  # Setting tools
  yScaler = RangeScaler_C()
  yScaler.Load(modelPN / f'yScaler.pck')
  annModel = load_model(modelPN / f'annModel.keras')

  # Getting data
  xTrain = np.load(trainingResultsPN / f'xTrain.npy')
  yTrain = np.load(trainingResultsPN / f'yTrain.npy')
  xTest  = np.load(trainingResultsPN / f'xTest.npy')
  yTest  = np.load(trainingResultsPN / f'yTest.npy')

  # Prediction
  startTime = time.time()
  yOut      = annModel.predict(xTest)
  endTime   = time.time()
  if setupOptions.verbose > 2:
    print('Testing time:', endTime - startTime)

  yTrainOut = annModel.predict(xTrain)

  # Data post-processing
  yTrainScaled    = yScaler.inverse_transform(yTrain)
  yTrainOutScaled = yScaler.inverse_transform(yTrainOut)
  yTestScaled     = yScaler.inverse_transform(yTest)
  yOutScaled      = yScaler.inverse_transform(yOut)

  wManager = WorkerManager_C()

  for dataPart, yReal, yEst in (('Train', yTrainScaled, yTrainOutScaled),
                                ('Test',  yTestScaled,  yOutScaled)):
    for i, outPar in enumerate(outParNames):
      plotDsc = {
        'outPN': trainingResultsPN,
        'graphBaseName': f'ParValueEvaluation_{outPar}_{dataPart}',
        'yTest': yReal[:,i],
        'yOut' : yEst[:,i],
        'imgExt': setupOptions.imgExt,
      }
      wManager.Run(GraphicWorkers.ParValueEvaluationWorker, (plotDsc,))

      plotDsc['graphBaseName'] = f'ParValueEvaluation_{outPar}_{dataPart}_Diff'
      wManager.Run(GraphicWorkers.ParValueEvaluationDiffWorker, (plotDsc,))

  wManager.Join()

def ParameterModelTesting(setupOptions):
  #=============================================
  trainingResultsPN = setupOptions.trainResultsPN
  modelPN = setupOptions.modelPN

  # Setting tools
  yScaler = RangeScaler_C()
  yScaler.Load(modelPN / f'yScaler.pck')
  annModel = load_model(modelPN / f'annModel.keras')

  # Getting data
  xTest  = np.load(trainingResultsPN / f'xTest.npy')

  # Prediction
  startTime = time.time()
  yOut      = annModel.predict(xTest)
  endTime   = time.time()
  if setupOptions.verbose > 2:
    print('Testing time:', endTime - startTime)

  # Data post-processing - transform data to its original range
  yOutScaled      = yScaler.inverse_transform(yOut)

  # Output
  # Load the header
  with open(modelPN / 'outParNames.pck', 'rb') as inF:
    outParNames = pickle.load(inF)
  # Save output as CSV
  WriteToCsv(yOutScaled, trainingResultsPN / f'parPredict.csv', outParNames)

def Predicting(setupOptions, runCfg, dataPool):
  modelPN = setupOptions.modelPN
  outPN = setupOptions.predResultsPN


  inSheetName = runCfg['trainingDataset']['inSheet']

  inParNames = dataPool[inSheetName]['colNames'][1:]  # skip the index


  inData = pd.DataFrame(dataPool[inSheetName]['data'],
                      index=dataPool[inSheetName]['index'],
                      columns=dataPool[inSheetName]['colNames'])
  xData = inData[inParNames].to_numpy()

  annModel = load_model(modelPN / f'annModel.keras')

  # Check whether the data have correct shape
  data_input_shape = (None,) + xData.shape[1:]
  model_input_shape = annModel.input_shape
  if data_input_shape != model_input_shape:
    raise ValueError(
      f'''The input shape of the prediction data differs from the input shape the model was trained on. Are you sure"
      you specified the prediction data correctly?
      Expected shape (Model input shape): {model_input_shape}
      Input shape of specified prediction dataset: {data_input_shape}'''
    )

  xScaler = RangeScaler_C()
  xScaler.Load(modelPN / f'xScaler.pck')
  yScaler = RangeScaler_C()
  yScaler.Load(modelPN / f'yScaler.pck')

  xPredict = xScaler.transform(xData)

  # Prediction
  startTime = time.time()
  yOut = annModel.predict(xPredict)
  endTime = time.time()

  if setupOptions.verbose > 2:
    print('Testing time:', endTime - startTime)

  # Data post-processing - transform data to its original range
  yOutScaled = yScaler.inverse_transform(yOut)


  # Output
  # Load the header
  with open(modelPN / 'outParNames.pck', 'rb') as inF:
    outParNames = pickle.load(inF)
  # Save output as CSV
  WriteToCsv(yOutScaled, outPN / f'parPredict.csv', outParNames)
      

# =======================================================================
# ===================== Resource Registration ===========================
# =======================================================================

# =======================================================================
# ===================== Test Functions ==================================
# =======================================================================

  
# =======================================================================
# ===================== MAIN ============================================
# =======================================================================

if __name__ == '__main__':
  pass
  

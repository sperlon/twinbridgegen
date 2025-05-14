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
                       

# =======================================================================
# ===================== Main Functions ==================================
# =======================================================================

def DataPreparation(setupOptions, runCfg, dataPool):
  #=================================================
  testSize          = setupOptions.testSize
  randomSeed        = runCfg['training']['randomSeed']
  outPN             = pathlib.Path(setupOptions.outPN, setupOptions.datasetDir, setupOptions.expTag)
  trainingResultsPN =  pathlib.Path(outPN, setupOptions.trainingResultsDir)
  modelPN           = pathlib.Path(outPN, setupOptions.modelDir)


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
  xScaler.Save(modelPN / f'xScaler.pck')
  yScaler.Save(modelPN / f'yScaler.pck')

  wManager.Join()

def ParameterModelTraining(setupOptions, runCfg):
  #==============================================
  outPN      = pathlib.Path(setupOptions.outPN, setupOptions.datasetDir, setupOptions.expTag)
  trainingResultsPN = pathlib.Path(outPN, setupOptions.trainingResultsDir)
  modelPN = pathlib.Path(outPN, setupOptions.modelDir)
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
    wManager = WorkerManager_C()

    plotDsc = {
    'outPN':          trainingResultsPN,
    'graphBaseName':  f'annModel_TrainHistory.png',
    'history':        history.history,
    }
    wManager.Run(GraphicWorkers.TrainingHistoryWorker, (plotDsc,))

    wManager.Join()

def ParameterModelEvaluation(setupOptions, runCfg):
  # ===============================================
  inPN  = pathlib.Path(setupOptions.outPN, setupOptions.datasetDir, setupOptions.expTag)
  outPN = pathlib.Path(setupOptions.outPN, setupOptions.datasetDir, setupOptions.expTag)
  trainingResultsPN = pathlib.Path(outPN, setupOptions.trainingResultsDir)
  modelPN = pathlib.Path(outPN, setupOptions.modelDir)

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

def ParameterModelTesting(setupOptions, runCfg):
  #=============================================
  inPN  = pathlib.Path(setupOptions.outPN, setupOptions.datasetDir, setupOptions.expTag)
  outPN = pathlib.Path(setupOptions.outPN, setupOptions.datasetDir, setupOptions.expTestTag)
  trainingResultsPN = pathlib.Path(outPN, setupOptions.trainingResultsDir)
  modelPN = pathlib.Path(outPN, setupOptions.modelDir)

  with open(modelPN / 'outParNames.pck', 'rb') as inF:
    outParNames = pickle.load(inF)

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

  # Data post-processing
  yOutScaled      = yScaler.inverse_transform(yOut)

  # Output
  with open(trainingResultsPN / f'parPredict.csv', 'wt') as csvF:
    parWriter = csv.writer(csvF, delimiter=',')
    parWriter.writerow(outParNames)
    for yRow in yOutScaled:
      parWriter.writerow(yRow)

def Predicting(dataPool, paths, verbose, runCfg):
  modelPN = paths['modelPN']
  outPN = paths['predictionPN']


  inSheetName = runCfg['trainingDataset']['inSheet']

  inParNames = dataPool[inSheetName]['colNames'][1:]  # skip the index


  inData = pd.DataFrame(dataPool[inSheetName]['data'],
                      index=dataPool[inSheetName]['index'],
                      columns=dataPool[inSheetName]['colNames'])
  xData = inData[inParNames].to_numpy()

  xScaler = RangeScaler_C()
  xScaler.Load(modelPN / f'xScaler.pck')
  yScaler = RangeScaler_C()
  yScaler.Load(modelPN / f'yScaler.pck')

  xPredict = xScaler.transform(xData)

  annModel = load_model(modelPN / f'annModel.keras')

  # Prediction
  startTime = time.time()
  yOut = annModel.predict(xPredict)
  endTime = time.time()

  if verbose > 2:
    print('Testing time:', endTime - startTime)

  # Data post-processing
  yOutScaled = yScaler.inverse_transform(yOut)



  with open(modelPN / 'outParNames.pck', 'rb') as inF:
    outParNames = pickle.load(inF)

  # Output
  with open(outPN / f'parPredict.csv', 'wt') as csvF:
    parWriter = csv.writer(csvF, delimiter=',')
    parWriter.writerow(outParNames)
    for yRow in yOutScaled:
      parWriter.writerow(yRow)

  pass
      

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
  

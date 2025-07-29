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
import ast

from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Reshape
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.utils import plot_model
from LibPy.ParallelWorkers import WorkerManager_C
from LibPy import GraphicWorkers
from LibPy.Model import DenseModel, MultiChannelModelC, DividedModel
from typing import Any, List, Dict, Union, Tuple
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


def get_column_indices(index_string: str, column_names: list[str]) -> Tuple[Union[List[int], List[Any]], List[int]]:
  """
  Parses a string containing a list (which can be nested) of column indices
  or column names and returns a tuple containing:
  1. A list of integer column indices with the same nested structure as input
  2. A list specifying the count of indices in each sublist/element

  Args:
      index_string: A string representation of a list, containing either
                    integer indices (e.g., "[[0, 2], 3]") or string column
                    names (e.g., "[['strain1', 'strain2'], 'temp']").
      column_names: A list of all column names in order.

  Returns:
      A tuple containing:
      - converted_indices: List of integer column indices with same nested structure
      - counts: List of integers specifying count of indices in each sublist/element
               (1 for individual elements, n for sublists with n elements)

  Raises:
      ValueError: If the string format is invalid, a column name is not found,
                  or the list contains unsupported types.
  """
  try:
    # Safely evaluate the string to a Python list.
    # ast.literal_eval is safer than eval() as it only processes literals.
    parsed_list = ast.literal_eval(index_string)
  except (ValueError, SyntaxError):
    raise ValueError("Invalid input string format. It should be a list in string format.")

  if not isinstance(parsed_list, list):
    raise ValueError("Input string must represent a list.")

  # Create a mapping of column names to their indices for faster lookup.
  name_to_index = {name: i for i, name in enumerate(column_names)}

  # Use a recursive helper function to convert while preserving structure.
  converted_indices = _convert_preserving_structure(parsed_list, name_to_index)

  # Get the counts for each sublist/element
  counts = _get_counts(parsed_list)

  return converted_indices, counts


def _get_counts(item: Any) -> List[int]:
  """
  Returns a list of counts for each element/sublist in the input structure.
  For individual elements, returns 1. For sublists, returns their length.
  """
  if isinstance(item, list):
    counts = []
    for sub_item in item:
      if isinstance(sub_item, list):
        counts.append(len(sub_item))
      else:
        counts.append(1)
    return counts
  else:
    # If the top-level item is not a list, return [1]
    return [1]


def _convert_preserving_structure(item: Any, name_to_index: Dict[str, int]) -> Any:
  """
  Recursively processes an item, which can be a list, int, or string,
  and returns the same structure with column names converted to indices.
  """
  if isinstance(item, int):
    return item

  if isinstance(item, str):
    if item not in name_to_index:
      raise ValueError(f"Column name '{item}' not found in the provided column list.")
    return name_to_index[item]

  if isinstance(item, list):
    return [_convert_preserving_structure(sub_item, name_to_index) for sub_item in item]

  raise ValueError(f"Unsupported type in list: {type(item).__name__}")

    
def TrainModel(setupOptions, runCfg, xN, yN, xTrain, yTrain):
  #==================================================================
  modelType = setupOptions.modelType
  epochN          = setupOptions.epochN
  verbose         = setupOptions.verbose
  learningRate    = runCfg['training']['learningRate']
  loss            = runCfg['training']['loss']
  metrics         = runCfg['training']['metrics'].split(',')
  validationSplit = runCfg['training']['validationSplit']

  modelPN = setupOptions.modelPN
  with open(modelPN / 'outParNames.pck', 'rb') as inF:
    outParNames = pickle.load(inF)

  # ANN model
  # determine the model type
  if modelType == "denseModel":
    layers = [int(s) for s in setupOptions.denseLayers.split(',')]
    annModel = DenseModel(layers, xN, yN)
  elif modelType == "multiChannelModel":
    layers = ast.literal_eval(setupOptions.multiLayers)
    inp_slices, inp_dims = get_column_indices(setupOptions.multiInput, outParNames)
    annModel = MultiChannelModelC(layers, inp_dims, inp_slices, yN)
  else:
    raise Exception(f"Unknown model type: '{modelType}'")

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
  xTrain = np.load(trainingResultsPN / f'xTrain.npy')
  yTrain = np.load(trainingResultsPN / f'yTrain.npy')
  xN = len(xTrain[0])
  yN = len(yTrain[0])
  annModel, history = TrainModel(setupOptions, runCfg, xN, yN, xTrain, yTrain)
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
  

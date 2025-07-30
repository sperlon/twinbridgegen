# Author:  Radek Marik, radek.marik@fel.cvut.cz
# Created: Dec 21, 2024
# Purpose: function estimation of input/output relations
# To create EXE: pyinstaller twinBridgePar.spec

# Copyright (c) 2024, Radek Marik, FEE CTU, Prague
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
import multiprocessing # because of an issue with PyInstaller
import argparse
import os.path
import pathlib
import tomllib
from ndicts import NestedDict
from Config import appDefaultValues
from LibPy.InputDataReader import ReadInputData
from LibPy.ParameterLearning import (DataPreparation, ParameterModelTraining,
                                     ParameterModelEvaluation, ParameterModelTesting, Predicting, InverseAnalysis)

Set_C = set

# =======================================================================
# ===================== Support Structures ==============================
# =======================================================================

if 0: print('appDefaultValues', appDefaultValues)

# =======================================================================
# ===================== Support Functions ===============================
# =======================================================================

def CommandLine(runCfgFN, runCfg):
  #===============================
  # Basic command line pars: -prepareData -isTraining isPrediction

  # Command line default values
  rootInPN  = pathlib.Path(runCfg["appPaths"]["rootInPN"])
  rootOutPN = pathlib.Path(runCfg["appPaths"]["rootOutPN"])

  expTag    = runCfg["runIds"]["expTag"]
  datasetDir = runCfg["runIds"]["datasetDir"]
  projectDir = runCfg["runIds"]["projectDir"]
  inPN  = rootInPN / projectDir / datasetDir
  outPN = rootOutPN / projectDir / "Output"
  modelDir = runCfg["runIds"]["modelDir"]
  trainingResultsDir = runCfg["runIds"]["trainingResultsDir"]
  predictionResultsDir = runCfg["runIds"]["predictionResultsDir"]
  inverseResultsDir = runCfg["runIds"]["inverseResultsDir"]

  # Training dataset selection parameters
  trainingDataFN = runCfg["trainingDataset"]["dataFN"]
  trainingInPars = runCfg["trainingDataset"]["inPars"]
  trainingOutPars = runCfg["trainingDataset"]["outPars"]

  # Prediction dataset selection parameters
  predictionDataFN = runCfg["predictionDataset"]["dataFN"]
  predictionOutPars = runCfg["predictionDataset"]["inPars"]

  # Inverse analysis dataset selection parameters
  inverseAnalysisDataFN = runCfg["inverseAnalysisDataset"]["dataFN"]
  inverseAnalysisOutPars = runCfg["inverseAnalysisDataset"]["inPars"]

  # Paths
  outPN_final = pathlib.Path(outPN, datasetDir, expTag)

  trainResultsPN = pathlib.Path(outPN_final, trainingResultsDir)
  modelPN = pathlib.Path(outPN_final, modelDir)
  predResultsPN = pathlib.Path(outPN_final, predictionResultsDir)
  invAnResultsPN = pathlib.Path(outPN_final, inverseResultsDir)

  trainInPN = pathlib.Path(inPN, trainingDataFN)
  predInPN = pathlib.Path(inPN, predictionDataFN)
  invAnInPN = pathlib.Path(inPN, inverseAnalysisDataFN)


  # Create necessary directories
  trainResultsPN.mkdir(parents=True, exist_ok=True)
  modelPN.mkdir(parents=True, exist_ok=True)
  predResultsPN.mkdir(parents=True, exist_ok=True)
  invAnResultsPN.mkdir(parents=True, exist_ok=True)

  # Training parameters
  verbose  = runCfg["training"]["verbose"]
  testSize = runCfg["training"]["testSize"]
  epochN = runCfg["training"]["epochN"]
  learningRate = runCfg["training"]["learningRate"]

  # Model type
  modelType = runCfg["model"]["modelType"]

  # Dense model definition
  denseLayers = runCfg["denseModel"]["Layers"]

  # divided model definition
  dividedLayers = runCfg["dividedDenseModel"]["Layers"]

  inverseLearningRate = runCfg["dividedDenseModel"]["inverseParams"]["inverseLearningRate"]
  inverseNIter = runCfg ["dividedDenseModel"]["inverseParams"]["n_iter"]
  inverseBestOnly = runCfg["dividedDenseModel"]["inverseParams"]["return_best_only"]
  inverseLowerLim = runCfg["dividedDenseModel"]["inverseParams"]["lower_limit"]
  inverseUpperLim = runCfg["dividedDenseModel"]["inverseParams"]["upper_limit"]
  inverseToler = runCfg["dividedDenseModel"]["inverseParams"]["tolerance"]
  inverseMaxIter = runCfg["dividedDenseModel"]["inverseParams"]["max_iter"]
  inversePrintFreq = runCfg["dividedDenseModel"]["inverseParams"]["print_freq"]

  # multichannel model
  multiLayers = runCfg["multiChannelModel"]["Layers"]
  multiInput = runCfg["multiChannelModel"]["Inputs"]

  # Other parameters
  imgExt = runCfg["others"]["imgExt"]
  acknowledgement = runCfg["others"]["acknowledgement"]

  cliParser = argparse.ArgumentParser(description = 'Top level command line',
                                      epilog      = acknowledgement)

  cliParser.add_argument('-isTraining',    default = False,         help = 'Training phase', action="store_true")
  cliParser.add_argument('-isPrediction',  default = False,         help = 'Prediction phase', action="store_true")
  cliParser.add_argument('-isInverseAnalysis', default=False,    help='Prediction phase', action="store_true")
  cliParser.add_argument('-isTestingSubPart', default=False,        help = 'For development', action="store_true")
  cliParser.add_argument('-modelType',     default = modelType,     help = 'Specification of a model type that will be used')
  cliParser.add_argument('-runCfgFN',      default = runCfgFN,      help = 'Run configuration file (I)')
  cliParser.add_argument('-verbose',       default = verbose,       help = 'Verbose level (I)', type=int)
  cliParser.add_argument('-expTag',        default = expTag,        help = 'Experiment label (I)')
  cliParser.add_argument('-expTestTag',    default = expTag,        help = 'Experiment testing label (I)')
  cliParser.add_argument('-datasetDir',     default = datasetDir,     help = 'Dataset id (I)')
  cliParser.add_argument('-projectDir',     default = projectDir,     help = 'Project id (I)')
  cliParser.add_argument('-imgExt',        default = imgExt,        help = 'Extension of generated diagrams (O)')
  cliParser.add_argument('-inPN',          default = inPN,          help = 'Input data path (I)')
  cliParser.add_argument('-outPN',         default = outPN,         help = 'Project and output data path (O)')
  cliParser.add_argument('-dataFN',        default = trainingDataFN,        help = 'Input data path (I)')
  cliParser.add_argument('-inPars',        default = trainingInPars,        help = 'Input parameters (I)')
  cliParser.add_argument('-outPars',       default = trainingOutPars,       help = 'Output parameters (O)')
  cliParser.add_argument('-testSize',      default = testSize,      help = 'Part size of testing data (I)')
  cliParser.add_argument('-epochN',        default = epochN,        help = 'Number of learning epochs (I)', type=int)
  cliParser.add_argument('-learningRate',        default = learningRate,        help = 'Number of learning epochs (I)', type=float)
  cliParser.add_argument('-denseLayers',   default=denseLayers, help='Dense model layer sizes (I)')
  cliParser.add_argument('-dividedLayers', default=dividedLayers, help='Divided model layer sizes (I)')
  cliParser.add_argument('-inverseLearningRate', default=inverseLearningRate, help='Learning rate for inverse analysis.')
  cliParser.add_argument('-inverseNIter', default=inverseNIter, help='Number of iterations used to find the results.')
  cliParser.add_argument('-inverseBestOnly', default=inverseBestOnly, help='Whether return only one best iteration or every iteration in output file.')
  cliParser.add_argument('-inverseLowerLim', default=inverseLowerLim, help='Lower limit for the iteration')
  cliParser.add_argument('-inverseUpperLim', default=inverseUpperLim, help='Divided model layer sizes (I)')
  cliParser.add_argument('-inverseToler', default=inverseToler, help='Divided model layer sizes (I)')
  cliParser.add_argument('-inversePrintFreq', default=inversePrintFreq, help='Divided model layer sizes (I)')
  cliParser.add_argument('-inverseMaxIter', default=inverseMaxIter, help='Divided model layer sizes (I)')
  cliParser.add_argument('-predictionDataFN', default = predictionDataFN, help = 'Prediction data path (I)')
  cliParser.add_argument('-predictionOutPars', default = predictionOutPars, help = 'Output parameters (O)')
  cliParser.add_argument('-inverseAnalysisDataFN', default=inverseAnalysisDataFN, help='Prediction data path (I)')
  cliParser.add_argument('-inverseAnalysisOutPars', default=inverseAnalysisOutPars, help='Output parameters (O)')
  cliParser.add_argument('-multiLayers', default=multiLayers, help='Multichannel model layer specification (I)')
  cliParser.add_argument('-multiInput', default=multiInput, help='Multichannel input layer specification (I)')
  cliParser.add_argument('-modelDir', default = modelDir, help = 'Model directory name (O)')
  cliParser.add_argument('-trainResultsDir', default=trainingResultsDir, help='Training results directory name (O)')
  cliParser.add_argument('-predResultsDir', default=predictionResultsDir, help='Prediction results directory name (O)')
  cliParser.add_argument('-trainResultsPN', default=trainResultsPN, help='Path where the training results are stored (O)')
  cliParser.add_argument('-modelPN', default=modelPN, help='Path where the trained model is stored (O)')
  cliParser.add_argument('-predResultsPN', default=predResultsPN, help='Path where the prediction results are stored (O)')
  cliParser.add_argument('-invAnResultsPN', default=invAnResultsPN, help='Path where the prediction results are stored (O)')
  cliParser.add_argument('-trainInPN', default=trainInPN, help='Path to the training dataset (I)')
  cliParser.add_argument('-predInPN', default=predInPN, help='Path to the prediction dataset (I)')
  cliParser.add_argument('-invAnInPN', default=invAnInPN, help='Path to the prediction dataset (I)')


  options = cliParser.parse_args()
  if options.verbose > 3: print('options', options)
  return options

def SetOutPN(*subFolders):
  #=======================
  # Open a necessary output infrastructure
  outPN = os.path.join(*subFolders)
  if not os.path.exists(outPN):
    os.makedirs(outPN)
  return outPN

# =======================================================================
# ===================== Classes =========================================
# =======================================================================

# =======================================================================
# ===================== Main Functions ==================================
# =======================================================================

def TwinBridgeGen(appDefaultValues):
  #=================================
  runCfgFN = appDefaultValues["appPaths"]["runCfgFN"]
  initOptions  = CommandLine(runCfgFN, appDefaultValues)

  # Application run configuration file parameters
  appRunValues = dict()
  cfgPath      = pathlib.Path(__file__) / initOptions.runCfgFN
  if cfgPath.exists():
    with cfgPath.open(mode="rb") as fp:
      appRunValues = tomllib.load(fp)

  defaultCfg = NestedDict(appDefaultValues)
  runCfg     = NestedDict(appRunValues)
  runCfg.update(defaultCfg)

  setupOptions = CommandLine(initOptions.runCfgFN, runCfg)

  SetOutPN(setupOptions.outPN)
  # added only for debugging
  if setupOptions.isTestingSubPart:
    dataPool = ReadInputData(setupOptions.trainInPN, setupOptions.trainResultsPN, setupOptions.verbose, runCfg)
    DataPreparation(setupOptions, runCfg, dataPool)

  if setupOptions.isTraining:
    dataPool = ReadInputData(setupOptions.trainInPN, setupOptions.trainResultsPN, setupOptions.verbose, runCfg)
    DataPreparation(setupOptions, runCfg, dataPool)
    ParameterModelTraining(setupOptions, runCfg)
    ParameterModelEvaluation(setupOptions)
    ParameterModelTesting(setupOptions)

  if setupOptions.isPrediction:
    dataPool = ReadInputData(setupOptions.predInPN, setupOptions.predResultsPN, setupOptions.verbose, runCfg, mode="prediction")
    Predicting(setupOptions, runCfg, dataPool)

  if setupOptions.isInverseAnalysis:
    dataPool = ReadInputData(setupOptions.invAnInPN, setupOptions.invAnResultsPN, setupOptions.verbose, runCfg, mode="inverseAnalysis")
    InverseAnalysis(setupOptions, runCfg, dataPool)

  
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
  import tensorflow

  print(f"TensorFlow Version: {tensorflow.__version__}")
  multiprocessing.freeze_support() # to avoid an issue with PyInstaller
  TwinBridgeGen(appDefaultValues)
  

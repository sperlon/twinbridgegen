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
                                     ParameterModelEvaluation, ParameterModelTesting)

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
  # Basic command line pars: -prepareData -isTraining isEvaluation -isTesting

  # Command line default values
  rootInPN  = pathlib.Path(runCfg["appPaths"]["rootInPN"])
  rootOutPN = pathlib.Path(runCfg["appPaths"]["rootOutPN"])

  expTag    = runCfg["runIds"]["expTag"]
  datasetId = runCfg["runIds"]["datasetId"]
  projectId = runCfg["runIds"]["projectId"]
  inPN  = rootInPN / projectId / datasetId
  outPN = rootOutPN / projectId / "Output"

  # Dataset selection parameters
  dataFN = runCfg["dataset"]["dataFN"]
  inPars = runCfg["dataset"]["inPars"]
  outPars = runCfg["dataset"]["outPars"]

  # Training parameters
  verbose  = runCfg["training"]["verbose"]
  testSize = runCfg["training"]["testSize"]
  denseLayers = runCfg["training"]["denseLayers"]
  epochN = runCfg["training"]["epochN"]

  # Other parameters
  imgExt = runCfg["others"]["imgExt"]
  acknowledgement = runCfg["others"]["acknowledgement"]

  cliParser = argparse.ArgumentParser(description = 'Top level command line',
                                      epilog      = acknowledgement)
  cliParser.add_argument('-prepareData',   default = False,         help = 'Prepare data for training and testing', action="store_true")
  cliParser.add_argument('-isTraining',    default = False,         help = 'Training phase', action="store_true")
  cliParser.add_argument('-isEvaluation',  default = False,         help = 'Evaluation phase', action="store_true")
  cliParser.add_argument('-isTesting',     default = False,         help = 'Testing phase', action="store_true")
  cliParser.add_argument('-runCfgFN',      default = runCfgFN,      help = 'Run configuration file (I)')
  cliParser.add_argument('-verbose',       default = verbose,       help = 'Verbose level (I)', type = int)
  cliParser.add_argument('-expTag',        default = expTag,        help = 'Experiment label (I)')
  cliParser.add_argument('-expTestTag',    default = expTag,        help = 'Experiment testing label (I)')
  cliParser.add_argument('-datasetId',     default = datasetId,     help = 'Dataset id (I)')
  cliParser.add_argument('-projectId',     default = projectId,     help = 'Project id (I)')
  cliParser.add_argument('-imgExt',        default = imgExt,        help = 'Extension of generated diagrams (O)')
  cliParser.add_argument('-inPN',          default = inPN,          help = 'Input data path (I)')
  cliParser.add_argument('-outPN',         default = outPN,         help = 'Project and output data path (O)')
  cliParser.add_argument('-dataFN',        default = dataFN,        help = 'Input data path (I)')
  cliParser.add_argument('-inPars',        default = inPars,        help = 'Input parameters (I)')
  cliParser.add_argument('-outPars',       default = outPars,       help = 'Output parameters (O)')
  cliParser.add_argument('-testSize',      default = testSize,      help = 'Part size of testing data (I)')
  cliParser.add_argument('-denseLayers',   default = denseLayers,   help = 'Dense layer sizes (I)')
  cliParser.add_argument('-epochN',        default = epochN,        help = 'Number of learning epochs (I)', type = int)

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
  
  if setupOptions.prepareData:
    dataPool = ReadInputData(setupOptions, runCfg)
    DataPreparation(setupOptions, runCfg, dataPool)
  
  if setupOptions.isTraining:
    ParameterModelTraining(setupOptions, runCfg)

  if setupOptions.isEvaluation:
    ParameterModelEvaluation(setupOptions, runCfg)

  if setupOptions.isTesting:
    ParameterModelTesting(setupOptions, runCfg)
  
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
  multiprocessing.freeze_support() # to avoid an issue with PyInstaller
  TwinBridgeGen(appDefaultValues)
  

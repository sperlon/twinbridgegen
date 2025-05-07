#!/usr/bin/env python
# -*- coding: utf-8 -*- 
'''
Created on December 04, 2022

@author: Radek Marik
'''

# Author:  Radek Marik, RMarik@gmail.com
# Created: December 04, 2022
# Purpose: graphic workers for parallel processing

# Copyright (c) 2022, Radek Marik, FEE CTU, Prague
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
import random, os.path
import numpy as np
import gc # memory leak avoidance for matplotlib
import multiprocessing as mp
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm as cm
from matplotlib.colors import LogNorm
Set_C = set

# =======================================================================
# ===================== Support Structures ==============================
# =======================================================================

# =======================================================================
# ===================== Support Functions ===============================
# =======================================================================

# =======================================================================
# ===================== Classes =========================================
# =======================================================================

# =======================================================================
# ===================== Main Functions ==================================
# =======================================================================

def SimulationsWorker(sema, plotSpec, fName, xyCurves):
  #====================================================
  outPN   = plotSpec['outPN']
  par2idx = {yPar: i for i, yPar in enumerate(plotSpec['yPars'])}
  yCurves = xyCurves.T
  for yLabel, yGroup in zip(plotSpec['yLabels'].split(','), plotSpec['yGroups'].split(',')):
    yData = yCurves[[i for yPar, i in par2idx.items() if yPar.startswith(yGroup)]]
    fig, ax = plt.subplots()
    xData = np.arange(len(yData[0]))
    for i in range(len(yData)):
      aCurve = yData[i]
      ax.plot(xData, aCurve, lw = 0.5)
    ax.set_xlabel('Simulation index')
    ax.set_ylabel(yLabel)
    fPath = os.path.join(outPN, f"{yLabel}_{fName}")
    plt.savefig(fPath)
    if 1: print('SimulationsWorker', fPath)
    plt.close(fig)

    yData = xyCurves[:,[i for yPar, i in par2idx.items() if yPar.startswith(yGroup)]]
    fig, ax = plt.subplots()
    xData = np.arange(len(yData[0]))
    for i in range(len(yData)):
      aCurve = yData[i]
      ax.plot(xData, aCurve, lw=0.5)
    ax.set_xlabel('Delay')
    ax.set_ylabel(yLabel)
    fPath = os.path.join(outPN, f"{yLabel}_time_{fName}")
    plt.savefig(fPath)
    if 0: print('SimulationsWorker', fPath)
    plt.close(fig)
  sema.release()

def ParValueEvaluationWorker(sema, plotSpec):
  #==========================================
  outPN         = plotSpec['outPN']
  graphBaseName = plotSpec['graphBaseName']
  yTest         = plotSpec['yTest']
  yOut          = plotSpec['yOut']
  imgExt        = plotSpec['imgExt']

  fig, ax = plt.subplots()
  sc = ax.scatter(yTest, yOut, s=4)
  ax.set_xlabel('Real Value')
  ax.set_ylabel('Estimated Value')
  plt.savefig(os.path.join(outPN, f'{graphBaseName}.{imgExt}'), dpi=600)
  plt.close(fig)
  sema.release()

def ParValueEvaluationDiffWorker(sema, plotSpec):
  #==============================================
  outPN         = plotSpec['outPN']
  graphBaseName = plotSpec['graphBaseName']
  yTest         = plotSpec['yTest']
  yOut          = plotSpec['yOut']
  imgExt        = plotSpec['imgExt']

  yDiff = yOut - yTest

  fig, ax = plt.subplots()
  sc = ax.scatter(yTest, yDiff, s=4)
  ax.set_xlabel('Real Value')
  ax.set_ylabel('Estimation Difference')
  plt.savefig(os.path.join(outPN, f'{graphBaseName}.{imgExt}'), dpi=600)
  plt.close(fig)
  sema.release()

def TrainingHistoryWorker(sema, plotDsc):
  #=====================================
  outPN          = plotDsc['outPN']
  graphBaseName  = plotDsc['graphBaseName']
  history        = plotDsc['history']

  stepLine = np.arange(len(history['loss']), dtype=int)
  fig, ax = plt.subplots()
  for hSlot in ('mse', 'val_mse', 'mae', 'val_mae', 'loss', 'val_loss'):#, 'accuracy', 'val_accuracy', ):
    if hSlot in history:
      ax.plot(stepLine, history[hSlot], label = hSlot.capitalize())
  ax.legend()
  ax.set_xlabel('Epoch')
  ax.set_ylabel('Metric')
  plt.savefig(os.path.join(outPN, graphBaseName))
  plt.close(fig)

  sema.release()

  
  
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
  

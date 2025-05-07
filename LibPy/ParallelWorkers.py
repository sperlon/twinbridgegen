#!/usr/bin/env python
# -*- coding: utf-8 -*- 
'''
Created on June 04, 2021

@author: Radek Marik
'''

# Author:  Radek Marik, RMarik@gmail.com
# Created: June 04, 2021
# Purpose: parallel processing strategies

# Copyright (c) 2021, Radek Marik, FEE CTU, Prague
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
import multiprocessing as mp
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

class WorkerManager_C(object):
  #---------------------------
  
  def __init__(self, restWorkers = 1):
    #=================================
    workerN           = mp.cpu_count() - restWorkers
    concurrency       = workerN
    self.sema         = mp.Semaphore(concurrency)
    self.allProcesses = []

  def Run(self, Worker, args, **kwargs):
    #===================================
    self.sema.acquire() 
    p = mp.Process(target = Worker, args = (self.sema,) + args, kwargs = kwargs) 
    self.allProcesses.append(p)
    p.start()
    
  def Join(self, clear = False):
    #===========================
    for p in self.allProcesses:
      p.join()
    if clear:
      self.allProcesses.clear()

# =======================================================================
# ===================== Main Functions ==================================
# =======================================================================

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
  

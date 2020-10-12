#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from keras.layers import Dense
from keras.models import Sequential
from keras.losses import binary_crossentropy, mean_squared_error
from keras.optimizers import SGD

from sklearn.metrics import roc_curve
from sklearn.preprocessing import MinMaxScaler, StandardScaler, KBinsDiscretizer, PowerTransformer, Binarizer
from math import sqrt, factorial

from generator import *
from sympy.solvers import solvers

from sympy import symbols

def find_fixpoints(a_1,b_1,c_1,d_1,e_1,a_2,b_2,c_2,d_2,e_2):
  x,y = symbols('x y', real=True)
  g1 = a_1*x + b_1*y + c_1*x**2 + d_1*x*y + e_1*y**2
  g2 = a_2*x + b_2*y + c_2*x**2 + d_2*x*y + e_2*y**2
  sol = solvers.solve((g1,g2),(x,y))
  sols = []
  for x in sol:
    sols.append((float(x[0]),float(x[1])))
  return sols

def analyze_eigvs(matrix):
  eigs = np.linalg.eigvals(matrix)
  realparts = []
  for x in eigs:
    if np.isreal(x): realparts.append(x)
    else: realparts.append(x.real)
  if realparts[0]*realparts[1]<0: return 'saddle'
  elif realparts[0]<0: return 'atractor'
  else: return 'repulsor' 

def types(a,b_vect):
  cases = {'saddle':[0],'atractor':[0],'repulsor':[0]}
  for b in b_vect:
    cases[analyze_eigvs(jacobian_matrix_calculator(a,b))][0] += 1
  return cases  

if __name__=='__main__':
  # ONLY ORDER 2 POR AHORA!
  data = {}
  names = ['x0','y0',
              'a_1','b_1','c_1','d_1','e_1',
              'a_2','b_2','c_2','d_2','e_2',
                                            ]
  for x in names: data[x] = []
  # d(x) = dt * (a_1 * x + b_1 * y + c_1 * x^2 + d_1 * x*y + e_1 * y^2) 
  # d(y) = dt * (a_2 * x + b_2 * y + c_2 * y*x + d_2 * y*x + e_2 * y^2) 
  #
  for q in range(30):
    a_,b_ = ode_parameters(), initial_condition()
    #
    b_vect = find_fixpoints(*a_)
    types_dict = types(a_, b_vect)
    #
    for n in ['saddle','atractor','repulsor']:
      data[n] = data.get(n,[]) + types_dict[n]
    param = b_.tolist() + a_.tolist() 
    for i,x in enumerate(names): data[x] += [param[i]]
  # 
  df = pd.DataFrame(data)
  df.to_csv('integrating.csv',index=False)
  print(df.head())

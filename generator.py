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

def generate_one(minimum=-100,maximum=100,size=1):
  return np.random.rand()*np.random.randint(minimum,maximum,size)

def initial_condition():
  x=10
  return generate_one(-x, x, 2)


def ode_parameters():
  """
  with ORDER 2 AND DIM 2 ONLY:
  we have...
  d(x_i)/dt = dim + (dim)!   possibilities
  hence...
  """
  x=10
  return generate_one(-x, x, 2*5)

def ode_parameters_B():
  """
  with ORDER 2 AND DIM 2 ONLY:
  we have...
  d(x_i)/dt = dim + (dim)!   possibilities
  hence...
  """
  x=10
  return generate_one(-x, x, 2*5+4)


def jacobian_matrix_calculator(params, points):
  """ 
  only for order 2!
  """
  dim = 2
  M_raw = params.reshape(2, -1)
  M = np.zeros(4).reshape(2,2)
  #
  for i in range(2):
    t = M_raw[i]
    for j in range(2):
      M[i] += t[j] # add "dim" lineal terms
      M[i] += t[2+j] * points[j] # add "dim" quadratic terms
      if j==1: M[i] += t[4+j-1]*points[j] # add second row 
  return M

def jacobian_matrix_calculator_B(params, points):
  """ 
  only for order 2!
  """
  dim = 2
  M_raw = params.reshape(2, -1)
  M = np.zeros(4).reshape(2,2)
  #
  for i in range(2):
    t = M_raw[i]
    for j in range(2):
      M[i] += t[j] # add "dim" lineal terms
      M[i] += 2*t[2+j] * points[j] # add "dim" quadratic terms
      if j==1: M[i] += 2*t[4+j-1]*points[j] # add second row
    M[i] += 6*t[-4]*points[0]**2 + 6*t[-3]*points[1]**2 #triple order
    M[i] += 6*t[-2]*points[0]**2 + 6*t[-3]*points[1]**2 #triple order 
     
  return M


def convergence_veredict(matrix):
  eigs = np.linalg.eigvals(matrix)
  realparts = []
  for x in eigs:
    if np.isreal(x): realparts.append(x)
    else: realparts.append(x.real)
  if max(realparts)>0: return 1
  else: return 0

def generate(a,b):
  return convergence_veredict(jacobian_matrix_calculator(a,b))

if __name__=='__main__':
  # ONLY ORDER 2 POR AHORA!
  data = {}
  names = ['x0','y0',
              'a_1','b_1','c_1','d_1','e_1',
              'a_2','b_2','c_2','d_2','e_2',
                                       'divergence',]
  for x in names: data[x] = []
  # d(x) = dt * (a_1 * x + b_1 * y + c_1 * x^2 + d_1 * x*y + e_1 * y^2) 
  # d(y) = dt * (a_2 * x + b_2 * y + c_2 * y*x + d_2 * y*x + e_2 * y^2) 
  #
  for q in range(30000):
    a_,b_ = ode_parameters(), initial_condition()
    c_ = generate(a_, b_)
    param = b_.tolist() + a_.tolist() + [c_]
    for i,x in enumerate(names): data[x] += [param[i]]
  # 
  df = pd.DataFrame(data)
  df.to_csv('2dorder2.csv',index=False)
  print(df.iloc[:,-1].value_counts())












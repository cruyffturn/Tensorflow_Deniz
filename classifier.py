#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Input:
      noFets:           scalar,      Number of features 
      regType:          scalar,      2 or 1 corresponds to L2 or L1 
      outDim:           scalar,     Number of classes in Y if binary classification should be 1      
      units:            N_layers,   Number of units of the models ex: [0] corresponds to logistic reg. and [100,100] a                                                                              MLP with 2 hidden layers with 100 units each.
      


"""
#%% Imports
import os 
currPath = os.getcwd()
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pickle
import numpy as np
from helper2 import getTrainTest
from sklearn.preprocessing import StandardScaler

noFets = 7
regType = 2
outDim

strdz = 1

units = np.array([100,100])

# %% Tensor Definiitons
from networks import MLP

tf.reset_default_graph()

x_in = tf.placeholder( dtype = tf.float64, shape = [None,noFets])
y_in = tf.placeholder( dtype = tf.int64, shape = [None])
lmbdaIn = tf.placeholder( dtype = tf.float64, shape = [])

if regType == 1:
    reg = tf.contrib.layers.l1_regularizer( lmbdaIn)
else:
    reg = tf.contrib.layers.l2_regularizer( 2*lmbdaIn)


with tf.variable_scope("Model"):
    
    logits = MLP( x_in, units, tf.nn.relu, outDim, reg = reg)

if outDim != 1:
    loss = tf.losses.sparse_softmax_cross_entropy( 
            labels = y_in, logits = logits)
else:
    loss = tf.losses.sigmoid_cross_entropy(
            labels = y_in, logits = logits)

totalPar = 0
for iTens in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope ='Model'):
    if 'kernel:' in iTens.name or 'bias:' in iTens.name:
        totalPar += np.prod( iTens.shape.as_list())

lossReg = tf.losses.get_regularization_loss() / totalPar

lr_in = tf.placeholder( dtype = tf.float64, shape = [])
optimus = tf.train.AdamOptimizer( learning_rate = lr_in)

train_op = optimus.minimize( loss + lossReg)
pred = tf.math.argmax( logits, 1)
acc = tf.math.reduce_mean( tf.cast( tf.math.equal( pred, y_in), tf.float32))        


#Summaries

sumLoss = tf.summary.scalar('Loss1', loss)
tf.summary.scalar('Loss2', lossReg/lmbdaIn)
sumAcc = tf.summary.scalar('Acc', acc)
tf.summary.scalar('lmbda', lmbdaIn)
#tf.summary.histogram('Weights',W)

sumStd = 0
for iTens in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope ='Model'):
    if 'kernel:' in iTens.name:
        print( iTens.name)
        sumStd += tf.keras.backend.std( tf.layers.flatten( iTens))

tf.summary.scalar( 'std', sumStd)

merged = tf.summary.merge_all()
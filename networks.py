#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:39:49 2019

@author: EceKoyuncu
"""
import tensorflow as tf

def hLayer( x, units, activation, reg):
        
    #Only hidden layers no logit at the end
    
    if units[0] != 0:
        
        for i in range( units.shape[0]):
                
            if i == 0:
                h = tf.layers.dense( inputs = x, 
                                     units = units[i],
                                     activation = activation,
                                     kernel_regularizer = reg,
                                     bias_regularizer = reg                                     
                                     )    
            else:
                h = tf.layers.dense( inputs = h, 
                                     units = units[i],
                                     activation = activation,
                                     kernel_regularizer = reg,
                                     bias_regularizer = reg
                                     )
    else:
        
        print( 'Err: Not intended for units[0] = 0')
    return h

def MLP( x, units, activation, outDim, use_bias = True, init = None, reg = None):
    
    if reg is None:
        
        reg = tf.contrib.layers.l1_regularizer( 0.)     #Turns of the regularization    
    
    if units[0] != 0:
        
        h = hLayer( x, units, activation, reg)

    else:
        h = x 

    if init is None:           
        logits = tf.layers.dense( inputs = h, 
                             units = outDim,
                             use_bias = use_bias,
                             kernel_regularizer = reg,
                             bias_regularizer = reg
                             )     
    else:
        logits = tf.layers.dense( inputs = h, 
                             units = outDim,
                             use_bias = use_bias,
                             kernel_initializer = init,
                             kernel_regularizer = reg,
                             bias_regularizer = reg
                             )     
    return logits
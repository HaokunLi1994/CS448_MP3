#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 16:41:47 2018

@author: Haokun Li
"""

import numpy as np

def load_data(path):
    """ Load a data set.
    
    Args:
        path(str): path for a data file
    Returns:
        X(np.array): 3-D array, eg. (2436, 32, 32)
        y(np.array): 1-D array, eg. (2436,)
    """
    myfile = open(path, 'r')
    
    NEW_PIC = True
    X_temp = []
    X = []
    y = []
    
    for line in myfile:
        if NEW_PIC == True:
            X_temp = []
            NEW_PIC = False
        
        if len(line) > 3:
            row = list(line.strip())
            row = [int(x) for x in row]
            X_temp.append(row)
        else:
            y_class = int(line.strip())
            y.append(y_class)
            X.append(X_temp)
            NEW_PIC = True
    
    X = np.array(X, np.int)
    y = np.array(y, np.int)
    myfile.close()
    return X, y

def accuracy(label, pred):
    """ Compute accuracy by comparing predicted labels and 
    true labels.
    
    Args:
        label(np.array): 1-D, true labels
        pred(np.array): 1-D, predicted labels
    Returns:
        acc(float): accuracy
    """
    correct = np.sum(label==pred)
    return correct / len(label)
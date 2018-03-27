#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 22:01:11 2018

@author: Haokun Li
"""

import time
import itertools
import utils
from naive_bayes import NaiveBayes
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def heat_map(digit, model):
    """ Plot heat map of likelihood
    
    Args:
        digit_arr(list): list of digit pairs
        model(NaiveBayes)
    Returns:
        (None)
    """
    # Initialize likelihood dict
    model.compute_likelihood(model.X, model.y)
    
    # Initialize heat map values
    N, height, width = model.X.shape
    myplot = np.zeros(shape=(height, width))
    
    pos = itertools.product(range(height), range(width))
    for position in pos:
        myplot[position[0]][position[1]] = model.likelihood[digit][position][1]
    
    # Plot
    sns.heatmap(myplot)
    pass

def odds_map(digit1, digit2, model):
    """ Plot heat map of log odds ratios
    
    Args:
        digit1(int)
        digit2(int)
        model(NaiveBayes)
    Returns:
        (None)
    """
    # Initialize likelihood dict
    model.compute_likelihood(model.X, model.y)
    
    # Initialize heat map values
    N, height, width = model.X.shape
    map1 = np.zeros(shape=(height, width))
    map2 = np.zeros(shape=(height, width))
    
    pos = itertools.product(range(height), range(width))
    for position in pos:
        map1[position[0]][position[1]] = model.likelihood[digit1][position][1] + model.laplace
        map2[position[0]][position[1]] = model.likelihood[digit2][position][1] + model.laplace
        
    odds_map = np.divide(map1, map2)
    odds_map = np.log(odds_map)
    
    # Plot
    sns.heatmap(odds_map)
    pass

if __name__ == '__main__':
    start = time.time()
    
    # Global variables
    TRAIN_PATH = ('digitdata/optdigits-orig_train.txt')
    TEST_PATH = ('digitdata/optdigits-orig_test.txt')
    
    # Load data sets
    X_train, y_train = utils.load_data(TRAIN_PATH)
    X_test, y_test = utils.load_data(TEST_PATH)
    
    # Initialize model
    model = NaiveBayes(laplace=0.1)
    
    # Fit model
    model.fit(X_train, y_train)
    
    # Predict
    pred_matrix, pred = model.predict(X_test)
    
    # Output
    end = time.time()
    acc = utils.accuracy(y_test, pred)
    print('Time used: {0:.3f} second(s)'.format(end - start))
    print('Smoothing: {0}'.format(model.laplace))
    print('Accuracy is {0:.3f}\n'.format(acc))
    print('Confusion matrix:')
    print(pd.DataFrame(utils.confusion_matrix(y_test, pred, model)), '\n')
    print('Highest tokens:')
    print(np.argmax(pred_matrix, axis=0), '\n')
    print('Lowest tokens:')
    print(np.argmin(pred_matrix, axis=0))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 19:00:53 2018

@author: Haokun Li
"""

import time
import utils
import numpy as np
import pandas as pd
from sklearn.svm import SVC

if __name__ == '__main__':
    start = time.time()
    
    # Global variables
    TRAIN_PATH = ('digitdata/optdigits-orig_train.txt')
    TEST_PATH = ('digitdata/optdigits-orig_test.txt')
    
    # Load data sets
    X_train, y_train = utils.load_data(TRAIN_PATH)
    X_train = np.reshape(X_train, (-1, 1024))
    X_test, y_test = utils.load_data(TEST_PATH)
    X_test = np.reshape(X_test, (-1, 1024))
    
    # Initialize model
    model = SVC(C=10, tol=1e-6)
    
    # Fit model
    model.fit(X_train, y_train)
    
    # Predict
    pred = model.predict(X_test)
    
    # Output
    end = time.time()
    acc = utils.accuracy(y_test, pred)
    print('Time used: {0:.3f} second(s)'.format(end - start))
    print('Accuracy is {0:.3f}\n'.format(acc))
    print('Confusion matrix:')
    print(pd.DataFrame(utils.confusion_matrix(y_test, pred)), '\n')

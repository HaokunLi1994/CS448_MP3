#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 16:24:08 2018

@author: Haokun Li
"""

import itertools
import numpy as np

class NaiveBayes(object):
    """ Naive bayes class."""
    def __init__(self, laplace=0.1, window=None):
        self.num_class = None
        self.labels = None
        self.possible_values = [0, 1]
        
        self.X = None
        self.y = None
        
        # Laplacian smoothing to avoid devision by 0
        self.laplace = laplace
        self.window = window
        
        # In the form of {label: prob}
        self.prior = dict()
        
        # In the form of {label: {position: {value: prob}}}
        self.likelihood = dict()
        pass
    
    def _compute_prior(self, y):
        """ Compute prior and update it.
        
        Args:
            y(np.array): 1-D, true labels
        Returns:
            (None)
        """
        labels = list(set(y))
        self.num_class = len(labels)
        self.labels = sorted(labels.copy())
        self.y = y
        
        for label in labels:
            prob = np.sum(y==label) / len(y)
            self.prior.update({label: prob})
        pass
    
    def fit(self, X, y):
        """ Fit training set.
        
        Args:
            X(np.array): 3-D, (#, height, width), features
            y(np.array): 1-D, true labels
        Returns:
            (None)
        """
        # Transform training data
        if self.window:
            self.possible_values = self.window.possible_values
            new_X = self.window.transform(X)
        else:
            new_X = X
        self.X = new_X
        self.y = y
        
        # Update self.prior
        self._compute_prior(y)
        pass
    
    def predict(self, test, method='prob'):
        """ Predict labels for testing set.
        
        Args:
            test(np.array): 3-D, (#, height, width), features
            method(str): 'prob' - return an array with probs
                         'label' - return an array with labels
        Returns:
            pred_matrix(np.array): 2-D, (#, # of labels), prob for each class
            pred(np.array): 1-D, (#,), predicted labels
        """
        # Transform test data
        if self.window:
            self.possible_values = self.window.possible_values
            new_test = self.window.transform(test)
        else:
            new_test = test
            
        N, height, width = new_test.shape
        pred_matrix = np.zeros(shape=(N, self.num_class))
        
        for i in range(N):
            for label in self.labels:
                # Select all pics of this class
                temp_class = self.X[np.where(self.y==label)]
                pos = itertools.product(range(height), range(width))
                log_likelihood = np.log(self.prior[label])
                
                # Iterate over all pixels
                for position in pos:
                    value = new_test[i][position[0]][position[1]]
                    temp_slice = temp_class[:, position[0], position[1]]
                    count = np.sum(temp_slice==value) + self.laplace
                    prob = count / (temp_slice.shape[0] + 
                                    len(self.possible_values) * self.laplace)
                    log_likelihood += np.log(prob)
                pred_matrix[i][label] = log_likelihood
        
        pred_matrix = np.array(pred_matrix)
        pred = np.argmax(pred_matrix, axis=1)        
        return pred_matrix, pred
    
    def compute_likelihood(self, X, y):
        """ Compute likelihood and update it.

        Args:
            X(np.array): 3-D, (#, height, width), features
            y(np.array): 1-D, true labels
        Returns:
            (None)
        """
        N, height, width = X.shape
        labels = list(set(y))
        
        for label in labels:
            temp_dict = dict()
            temp_class = X[np.where(y==label)]
            pos = itertools.product(range(height), range(width))
            for position in pos:
                temp_slice = temp_class[:, position[0], position[1]]
                temp_prob_dict = dict()
                for value in self.possible_values:
                    count = np.sum(temp_slice==value) + self.laplace
                    prob = count / (temp_slice.shape[0] + 2 * self.laplace)
                    temp_prob_dict.update({value: prob})                
                temp_dict.update({position: temp_prob_dict})                
            self.likelihood.update({label: temp_dict})
        pass

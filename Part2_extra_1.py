# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 14:10:11 2018

@author: jkell
"""

import DNN
import numpy as np
import utils
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy

lin = lambda inp: inp

class multiPerceptron:
    """ specific implentation of the neural network
    """
    def __init__(self):
        self.NNs = []
        for i in range(0,10):
            D = DNN.NeuralNetwork(1024, 1)
            D.add_hidden_layer(1, activation=lin, bias=False)
            D.generate_weights()
            self.NNs.append(D)
    
    def train(self,X, y, learningrate = 0.01):
        """ X is the feature vector
            y is the expected output """
        
        result = self.forward(X)
        
        if (result != y):
            # need to update weights
            toadd = np.reshape(learningrate * np.append(X, 1), (1025, 1))
            self.NNs[y].weights[0] = np.add(self.NNs[y].weights[0],toadd)
        
            self.NNs[result].weights[0] = np.subtract(self.NNs[result].weights[0],toadd)
        
    def forward(self,X):
        # predict the output based on features X
        results = np.zeros((10,))
        maxn = -999999999999
        maxr = 0
        for i in range(0,10):
            a = self.NNs[i].forward(X)
            results[i] = a
            if (results[i] > maxn):
                maxr = i
                maxn = results[i]
        
        return maxr
    
    def evaluate(self, X, y):
        """ Get a percent accuracy rating
        X is the input set
        y is the set of answers """
        correct = np.zeros((len(y),))
        for i in range(0,len(y)):
            r = P.forward(X[i,:])
            if r == y[i]:
                correct[i] = 1
        
        return np.mean(correct)*100
    
    
if __name__ == '__main__':
    print("Loading data...")
        # Global variables
    TRAIN_PATH = ('./digitdata/optdigits-orig_train.txt')
    TEST_PATH = ('./digitdata/optdigits-orig_test.txt')
    maxepochs=100
    
    # Load data sets
    X_train, y_train = utils.load_data(TRAIN_PATH)
    X_train = np.reshape(X_train, (2436, 1024))
    X_test, y_test = utils.load_data(TEST_PATH)
    X_test = np.reshape(X_test, (X_test.shape[0], 1024))
    print("Training data loaded.")
    
    print("Initializing network...")
    
    P = multiPerceptron()
    
    print("Done.")
    
    accuracy = []
    accuracy.append(P.evaluate(X_train, y_train))
    print("Starting accuracy:", accuracy[0],"%")
    
    print("Training...")
    
    # We're going to do this the slow way...
    for epoch in range(0, maxepochs):
        print("Epoch",epoch)
        for i in range(0,len(y_train)):
            P.train(X_train[i,:], y_train[i], 0.01*(0.99**epoch))
            
        accuracy.append(P.evaluate(X_train, y_train))
        if (accuracy[len(accuracy)-1] == 100):
            break
        
    print("Finished training after", len(accuracy)-1, "epochs.")
    
    plt.figure()
    plt.plot(accuracy)
    plt.title("Accuracy over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    
    print("Accuracy on test set:",P.evaluate(X_test, y_test),"%")
        
    pred=np.zeros(len(y_test),)
    
    for i in range(0, len(y_test)):
        pred[i] = P.forward(X_test[i,:])
        
    print(pd.DataFrame(utils.confusion_matrix(y_test, pred)), '\n')
    
    # Additional stuff for extra credit -- everything above this is duplicated from Part2_1.py
    
    plt.figure()
    
    for i in range(0,10):
        plt.subplot(5,2,i+1)
        weights = deepcopy(P.NNs[i].weights[0])    # grab the weights. there is only one hidden layer so just 1 weight vector
        weights = weights[0:len(weights)-1]   # throw out the last weight because it's a bias and we don't care about it
        weightimg = np.reshape(weights, (32,32))
        
        plt.imshow(weightimg, cmap='bwr')
        
        plt.title(str(i))
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off') # labels along the bottom edge are off
        
        plt.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            left='off',      # ticks along the bottom edge are off
            right='off',         # ticks along the top edge are off
            labelleft='off') # labels along the bottom edge are off
        plt.colorbar(orientation='vertical')   
        
        
     
    plt.tight_layout()
    plt.show()
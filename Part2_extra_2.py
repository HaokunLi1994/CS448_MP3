# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 11:31:48 2018

@author: jkell

Result:
    


"""

import DNN
import numpy as np
import utils
import matplotlib.pyplot as plt
import pandas as pd

lin = lambda inp: inp

def softmax(NNs, x, c, sumstart=0, idxstart=0):
    #sumstart is for optimization
    summ = sumstart
    for i in range(idxstart,c+1):
        summ += np.exp(NNs[i].forward(x))
    
    return np.exp(NNs[c].forward(x))/summ, summ

class multiPerceptron:
    """ specific implentation of the neural network
    differentiable learning and weight updates
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
            # 0.3699 = 1/2*(tanh(1)+1)*(1-tanh(1)^2)
            toadd = np.reshape(learningrate * 0.3699 * np.append(X, 1), (1025, 1))
            self.NNs[y].weights[0] = np.add(self.NNs[y].weights[0],toadd)
        
            self.NNs[result].weights[0] = np.subtract(self.NNs[result].weights[0],toadd)
        
    def forward(self,X):
        # predict the output based on features X
        results = np.zeros((10,))
        maxn = -999999999999
        maxr = 0
        summ=0
        for i in range(0,10):
            a,summ = softmax(self.NNs, X, i, summ, i)
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
            P.train(X_train[i,:], y_train[i], 0.1*(0.99**epoch))
            
        accuracy.append(P.evaluate(X_train, y_train))
        print(accuracy[len(accuracy)-1])
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
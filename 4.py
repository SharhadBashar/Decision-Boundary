#Import statements
import os
import csv
import os.path
import scipy.misc
import numpy as np
from os import path
import scipy.linalg as la
import numpy.random as npRand
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import minmax_scale
from matplotlib.collections import PatchCollection

class Q4():
    def __init__(self):
        self.dataLocation = 'dataset3.txt'
        self.w = np.random.rand(1, 2)[0]
        self.b = np.random.rand()
        self.epoch = 5000
        self.alpha = 1e-3
        self.cost = []
        #load
        self.genData()
        #run
        self.sgd()
        self.prediction()
        self.accuracy()
        #plot
        self.plotCost()
        self.plotData()
        
    def __str__(self):
        return 'Question 4'
    
    def genData(self):
        data = []
        y = []
        with open(self.dataLocation, 'r') as f:
            for line in f.readlines():
                line = line.rstrip().split(',')
                data.append([float(x) for x in line])
        data = minmax_scale(data, feature_range = (0, 1), axis = 0, copy = True)
        self.data = data
        return data
    
    def hypo(self, w, b, x):
        denom = 1 + np.exp(-(b + w[0] * x[0] + w[1] * x[1]))
        return 1 / denom
    
    def costFunc(self, w, b, x, y):
        sumCost = 0
        m = len(x)
        for i in range(m):
            h_theta_i = self.hypo(w, b, x[i])
            sumCost += y[i] * (np.log(h_theta_i)) + (1 - y[i]) * np.log(1 - h_theta_i)
        return -sumCost/m
    
    def gradW(self, w, b, x, y):
        return ((self.hypo(w, b, x) - y) * x)
    
    def gradB(self, w, b, x, y):
        return (self.hypo(w, b, x) - y)
    
    def sgd(self):
        w = self.w
        b = self.b
        for j in range(self.epoch):
            for i in range(len(self.data)):
                data = self.data
                np.random.shuffle(data)
                x = data[:, :2]
                y = data[:, 2].reshape((len(data), 1))
                
                w -= self.alpha * self.gradW(w, b, x[i], y[i])
                b -= self.alpha * self.gradB(w, b, x[i], y[i])
            self.cost.append(self.costFunc(w, b, x, y))
        self.w = w
        self.b = b
        print('Final thetas:', w[0], w[1], b[0])
        return [w, b]
    
    def prediction(self):
        pred = []
        features = self.data[:, :2]
        w = self.w
        b = self.b
        for i in range(len(features)):
            temp = features[i][0] * w[0] + features[i][1] * w[1] + b
            if (temp >= 0.5): pred.append(1)
            elif (temp <= 0.5): pred.append(0)
        self.pred = pred
        return pred
    
    def accuracy(self):
        actual = self.data[:, 2]
        pred = self.pred
        accuracy = 0
        for i in range(len(actual)):
            if (actual[i] == pred[i]):
                accuracy += 1
        print('Accuracy is:', accuracy/len(actual) * 100, '%')
        return accuracy / len(actual)
    
    def plotCost(self):
        fig = plt.figure(0, figsize = (10, 6))
        plt.plot(self.cost)
        plt.ylabel('Cost Function value')
        plt.xlabel('Epochs')
        plt.title('Cost Function along the epochs of the SGD')
        plt.show()
        
    def plotData(self):
        data = self.data
        X = data[:, : -1]
        y = data[:, -1]
        one = data[y == 1]
        zero = data[y == 0]
        
        x_values = [np.min(X[:, 0]), np.max(X[:, 1])]
        y_values = -(self.b + np.dot(self.w[0], x_values)) / self.w[1]
        
        fig = plt.figure(0, figsize = (10, 6))
        plt.scatter(one[:, 0], one[:, 1], c = 'red', marker = 'x', label = '1')
        plt.scatter(zero[:, 0], zero[:, 1], c = 'blue', marker = 'o', label = '0')
        plt.plot(x_values, y_values, label = 'Decision Boundary')
        plt.ylabel('x2')
        plt.xlabel('x1')
        plt.legend()
        plt.title('Data and class of the sample')
        plt.show()

Q4()
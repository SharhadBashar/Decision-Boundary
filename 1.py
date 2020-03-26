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

class Q1():
    def __init__(self):
        self.mean = np.zeros(2)
        self.A = [[1, 0], 
                  [0, 1]]
        self.B = [[2, -2], 
                  [-2, 3]]
        #run
        self.partA()
        self.partB()
        self.partC()
        #plot
        self.plot()
        

    def __str__(self):
        return 'Question 1'
    
    def cov(self, x, y):
        xMean, yMean = self.mean
        n = len(x)
        return sum((x - xMean) * (y - yMean)) / (n - 1)

    def covMat(self, data):
        return np.array([[self.cov(data[0], data[0]), self.cov(data[0], data[1])],
                     [self.cov(data[1], data[0]), self.cov(data[1], data[1])]])
    
    def eigen(self, cov):
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        eigenvalues = np.sqrt(eigenvalues)
        return[eigenvalues, eigenvectors]
    
    def ellipse(self, eigenvalues, eigenvectors):
        return Ellipse(xy = (self.mean[0], self.mean[1]), 
                       width = eigenvalues[0] * 2, 
                       height = eigenvalues[1] * 2, 
                       angle = -np.rad2deg(np.arccos(eigenvectors[0, 0])), 
                       lw = 5, fc = 'none', ec = 'red')
    
    def partA(self):
        data_A = npRand.multivariate_normal(self.mean, self.A, 1000).T
        data_B = npRand.multivariate_normal(self.mean, self.B, 1000).T
        self.data_A = data_A
        self.data_B = data_B
        return [data_A, data_B]
    
    def partB(self):
        eigenvalues_A, eigenvectors_A = self.eigen(self.A)
        eigenvalues_B, eigenvectors_B = self.eigen(self.B)
        ellipse_A = self.ellipse(eigenvalues_A, eigenvectors_A)
        ellipse_B = self.ellipse(eigenvalues_B, eigenvectors_B)
        self.ellipse_A = ellipse_A
        self.ellipse_B = ellipse_B
        return [ellipse_A, ellipse_B]
    
    def partC(self):
        cov_A = self.covMat(self.data_A)
        cov_B = self.covMat(self.data_B)
        self.cov_A = cov_A
        self.cov_B = cov_B
        print('Actual Covariance Martix:')
        print('A =', self.A)
        print()
        print('B =', self.B)
        print()
        print('Covariance matrices for part C:')
        print('A =', cov_A)
        print()
        print('B =', cov_B)
        return [cov_A, cov_B]
    
    def plot(self):
        fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 10))
        axes[0].add_artist(self.ellipse_A)
        axes[0].plot(self.data_A[0], self.data_A[1], 'o')
        axes[0].set_title('Samples for class A')
        
        axes[1].add_artist(self.ellipse_B)
        axes[1].plot(self.data_B[0], self.data_B[1], 'o')
        axes[1].set_title('Samples for class B')
        plt.show()

Q1()
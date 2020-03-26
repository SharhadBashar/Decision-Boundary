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

# Motivation for the classifier plots from: 
# https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html#sphx-glr-auto-examples-neighbors-plot-classification-py #
class Q2():
    def __init__(self):
        self.step = 0.1
        self.totalData = 3000
        
        #Class 1
        self.p_1 = 0.2
        self.mean_1 = [3, 2]
        self.cov_1 = [[1, -1], 
                     [-1, 2]]
        
        #Class 2
        self.p_2 = 0.7
        self.mean_2 = [5, 4]
        self.cov_2 = [[1, -1], 
                     [-1, 2]]
        
        #Class 3
        self.p_3 = 0.1
        self.mean_3 = [2, 5]
        self.cov_3 = [[0.5, 0.5], 
                     [0.5, 3]]
        
        self.prior = np.array([self.p_1, self.p_2, self.p_3])
        self.mean = np.array([self.mean_1, self.mean_2, self.mean_3])
        self.cov = np.array([self.cov_1, self.cov_2, self.cov_3])
        
        self.cmapLight = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
        self.cmapBold = ListedColormap(['darkorange', 'c', 'darkblue'])
        self.c = np.array([0] * int(self.totalData * self.p_1) + \
                          [1] * int(self.totalData * self.p_2) + \
                          [2] * int(self.totalData * self.p_3))
        #load
        self.genData()
        #run
        self.confusion()
        self.pEpsilon()
        #plot
        self.plotML()
        self.plotMAP()
    def __str__(self):
        return 'Question 2'
    
    def genData(self, totalData = 3000):
        data_1 = npRand.multivariate_normal(self.mean_1, self.cov_1, int(totalData * self.p_1))
        data_2 = npRand.multivariate_normal(self.mean_2, self.cov_2, int(totalData * self.p_2))
        data_3 = npRand.multivariate_normal(self.mean_3, self.cov_3, int(totalData * self.p_3))
        data = np.concatenate((data_1, data_2, data_3))
        self.data = data
        return data
    
    def gkxML(self, x, mean, cov):
        covInv = np.linalg.inv(cov)
        covNorm = np.linalg.norm(cov, ord = 1)
        gkx = []
        for i in range(len(x)):
            gkx.append(-0.5 * (x[i] - mean).T @ covInv @ (x[i] - mean) - 0.5 * np.log(covNorm))
        return np.array(gkx)
    
    def ML(self, data):
        gkx = []
        for i in range(len(self.prior)):
            gkx.append(self.gkxML(data, self.mean[i], self.cov[i]))
        predML = np.argmax(np.stack(gkx), axis = 0)
        return predML
    
    def gkxMAP(self, x, mean, cov, prior):
        covInv = np.linalg.inv(cov)
        covNorm = np.linalg.norm(cov, ord = 1)
        gkx = []
        for i in range(len(x)):
            gkx.append(-0.5 * (x[i] - mean).T @ covInv @ (x[i] - mean) - 0.5 * np.log(covNorm) + np.log(prior))
        return np.array(gkx)
    
    def MAP(self, data):
        gkx = []
        for i in range(len(self.prior)):
            gkx.append(self.gkxMAP(data, self.mean[i], self.cov[i], self.prior[i]))
        predMAP = np.argmax(np.stack(gkx), axis = 0)
        return predMAP
    
    def eigen(self, cov):
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        eigenvalues = np.sqrt(eigenvalues)
        return[eigenvalues, eigenvectors]
    
    def ellipse(self, mean, cov):
        eigenvalues, eigenvectors = self.eigen(cov)
        return Ellipse(xy = (mean[0], mean[1]), 
                       width = eigenvalues[0] * 2, 
                       height = eigenvalues[1] * 2, 
                       angle = -np.rad2deg(np.arccos(eigenvectors[0, 0])), 
                       lw = 5, fc = 'none', ec = 'red')
    
    def plotML(self):
        step = self.step
        data = self.data
        cmapLight = self.cmapLight
        cmapBold = self.cmapBold
        
        x1_min, x1_max = data[:, 0].min(), data[:, 0].max()
        x2_min, x2_max = data[:, 1].min(), data[:, 1].max()
        x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, step), np.arange(x2_min, x2_max, step))
        boundaryDataML = np.c_[x1.ravel(), x2.ravel()]

        pred = self.ML(data)
        boundaryPredML = self.ML(boundaryDataML)
        boundaryPredML = boundaryPredML.reshape(x1.shape)
        
        fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 10))
        plt.pcolormesh(x1, x2, boundaryPredML, cmap = cmapLight)
        plt.scatter(data[:, 0], data[:, 1], c = self.c, cmap = cmapBold, edgecolors = 'k', s = 20)
        plt.xlim(x1.min(), x1.max())
        plt.ylim(x2.min(), x2.max())
        plt.title('ML Classifier')
        axes.add_patch(self.ellipse(self.mean_1, self.cov_1))
        axes.add_patch(self.ellipse(self.mean_2, self.cov_2))
        axes.add_patch(self.ellipse(self.mean_3, self.cov_3))
        plt.show()
    
    def plotMAP(self):
        step = self.step
        data = self.data
        cmapLight = self.cmapLight
        cmapBold = self.cmapBold
        
        x1_min, x1_max = data[:, 0].min(), data[:, 0].max()
        x2_min, x2_max = data[:, 1].min(), data[:, 1].max()
        x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, step), np.arange(x2_min, x2_max, step))
        boundaryDataMAP = np.c_[x1.ravel(), x2.ravel()]

        pred = self.MAP(data)
        boundaryPredMAP = self.MAP(boundaryDataMAP)
        boundaryPredMAP = boundaryPredMAP.reshape(x1.shape)
        
        fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 10))
        plt.pcolormesh(x1,x2, boundaryPredMAP, cmap = cmapLight)
        plt.scatter(data[:, 0], data[:, 1], c = self.c, cmap = cmapBold, edgecolors = 'k', s = 20)
        plt.xlim(x1.min(), x1.max())
        plt.ylim(x2.min(), x2.max())
        plt.title('MAP Classifier')
        axes.add_patch(self.ellipse(self.mean_1, self.cov_1))
        axes.add_patch(self.ellipse(self.mean_2, self.cov_2))
        axes.add_patch(self.ellipse(self.mean_3, self.cov_3))
        plt.show()
        
    def confusion(self):
        trueLabel = self.c
        predML = self.ML(self.data)
        predMAP = self.MAP(self.data)
        confusionML = confusion_matrix(trueLabel, predML)
        confusionMAP = confusion_matrix(trueLabel, predMAP)
        self.confusionML = confusionML
        self.confusionMAP = confusionMAP
        print('ML Confusion matrix:')
        print(confusionML)
        print()
        print('MAP Confusion matrix:')
        print(confusionMAP)
        return [confusionML, confusionMAP]
    
    def pEpsilon(self):
        diagML = np.diag(self.confusionML)
        diagMAP = np.diag(self.confusionMAP)
        p_e_ML = (self.totalData - np.sum(diagML)) / self.totalData
        p_e_MAP = (self.totalData - np.sum(diagMAP)) / self.totalData
        self.p_e_ML = p_e_ML
        self.p_e_MAP = p_e_MAP
        print()
        print('P(e) for ML:', p_e_ML)
        print('P(e) for MAP:', p_e_MAP)
        return [p_e_ML, p_e_MAP]
Q2()
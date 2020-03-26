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

#code for dataset process codied from:
#https://github.com/pjreddie/mnist-csv-png
class processData():
    def __init__(self):
        self.csv("train-images-idx3-ubyte", "train-labels-idx1-ubyte", "train", 60000)
        
    def get_images(self, imgf, n):
        f = open(imgf, "rb")
        f.read(16)
        images = []

        for i in range(n):
            image = []
            for j in range(28*28):
                image.append(ord(f.read(1)))
            images.append(image)
        return images

    def get_labels(self, labelf, n):
        l = open(labelf, "rb")
        l.read(8)
        labels = []
        for i in range(n):
            labels.append(ord(l.read(1)))
        return labels

    def output_csv(self, images, labels, outf):
        o = open(outf, "w")
        for i in range(len(images)):
            o.write(",".join(str(x) for x in [labels[i]] + images[i]) + "\n")
        o.close()

    def csv(self, imgf, labelf, prefix, n):
        images = self.get_images(imgf, n)
        labels = self.get_labels(labelf, n)
        self.output_csv(images, labels, "mnist_%s.csv"%prefix)

class Q3():
    def __init__(self):
        self.fileLoc = 'mnist_train.csv'
        self.povTarget = 0.95
        self.target = 8
        self.mse = []
        self.reconstruction = []
        self.drange = np.arange(1, 785)
        self.dSkipRange = np.append(np.insert(np.arange(20, 780, 20), 0, 1), 784)
        self.d = [1, 10, 50, 250, 784]
        #load
        self.loadData()
        self.getNum()
        #run
        self.calcMean()
        self.sampleCov()
        self.POV()
        self.reconstructPCA()
        self.reconstructTarget()
        #plot
        self.plotC()
        self.plotD()
        self.plotE()
        
    def __str__(self):
        return 'Question 3'
    
    def loadData(self):
        file = open(self.fileLoc)
        data = np.array(list(csv.reader(file, delimiter = ',')), dtype = float)
        images = data[:, 1:]
        labels = data[:, 0]
        X = images.T
        self.images = images
        self.labels = labels
        self.X = X
        return [X, images, labels]
    
    def getNum(self):
        for i in range (self.labels.shape[0]):
            if self.labels[i] == self.target: 
                self.targetImage = self.images[i]
                return self.images[i]
    
    def calcMean(self):
        mean = np.sum(self.X.T, axis = 0) / self.X.shape[1]
        self.mean = mean
        return mean
    
    def sampleCov(self):
        X = self.X.T
        sampleCovX = (X - self.mean).T @ (X - self.mean) / (len(X) - 1)
        self.sampleCovX = sampleCovX
        return sampleCovX
    
    def eigen(self, cov):
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        return[eigenvalues, eigenvectors]
        
    def PCA(self, d):
        #1
        mean = self.calcMean()
        sampleCovX = self.sampleCov()
        #2
        X_Bar = self.X - mean.T.reshape(self.X.shape[0], 1)
        #3
        eigenvalues, eigenvectors = self.eigen(sampleCovX)
        #4
        sortEigenval = eigenvalues.argsort()[::- 1][: d]
        W = eigenvectors[:, sortEigenval]
        #5
        Y = W.T @ X_Bar
        self.Y = Y
        return [Y, W.T]
    
    def POV(self):
        eigenvalues, eigenvectors = self.eigen(self.sampleCovX)
        sortEigenval = np.sort(eigenvalues)[::-1]
        for d in range(1, self.X.shape[0] + 1):
            pov = np.sum(sortEigenval[: d], axis = 0) / np.sum(sortEigenval, axis = 0)
            if pov >= self.povTarget: 
                print('b) Suitable d for POV = 95% is:', d)
                return d
    
    def MSE(self, x_Hat):
        return np.sum((self.X - x_Hat) ** 2) / (self.X.shape[0] * self.X.shape[1])
    
    def reconstructPCA(self):
        for d in self.dSkipRange:
            Y, W = self.PCA(d)
            x_Hat = self.mean.reshape(W.shape[1], 1) + W.T @ Y
            self.mse.append(self.MSE(x_Hat))
        return x_Hat
    
    def reconstructTarget(self):
        targetImage = self.targetImage.reshape(self.X.shape[0], 1)
        for d in self.d:
            Y, W = self.PCA(d)
            imageRecon = self.mean.reshape(W.shape[1], 1) + W.T @ (W @ targetImage)
            self.reconstruction.append(imageRecon.reshape(28, 28))
    
    def plotC(self):
        fig = plt.figure(0, figsize = (10, 6))
        plt.plot(self.dSkipRange, self.mse)
        plt.title('c) MSE vs d')
        plt.xlabel('d')
        plt.ylabel('MSE')
        plt.show()
        
    def plotD(self):
        fig, axes = plt.subplots(nrows = 1, ncols = 6, figsize = (20, 10))
        for i in range(len(self.d)):
            axes[i].imshow(self.reconstruction[i])
            axes[i].set_title('d = ' + str(self.d[i]))
        axes[-1].imshow(self.targetImage.reshape(28, 28))
        axes[-1].set_title('Orignial ' + str(self.target))
        fig.suptitle('d) Reconstructed 8 for d = {1, 10, 50, 250, 784}')
        plt.show()
        
    def plotE(self):
        eigenvalues, eigenvectors = self.eigen(self.sampleCovX)
        fig = plt.figure(0, figsize = (10, 6))
        plt.plot(self.drange + 1, eigenvalues)
        plt.title('e) Eigenvalues vs d')
        plt.xlabel('d')
        plt.ylabel('Eigenvalues')
        plt.show()

if not path.exists('mnist_train.csv'): processData()
Q3()
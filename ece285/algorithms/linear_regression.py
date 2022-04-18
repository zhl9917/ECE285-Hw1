"""
Linear Regression model
"""

import numpy as np


class Linear(object):
    def __init__(self, n_class: int, lr: float, epochs: int, weight_decay: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # Initialize in train
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.weight_decay = weight_decay

    def train(self, X_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Train the classifier.

        Use the linear regression update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """

        N, D = X_train.shape
        self.w = weights

        # TODO: implement me

        #one hot encoding for y
        y_encode = np.zeros((N,weights.shape[0]))
        for i in range(weights.shape[0]):
            a = np.array(y_train==i)
            y_encode[:,i] = 2*a-1 #{-1,1} encoding
        
        for _ in range(self.epochs):
        #gradient decent function
            y_predict = self.w@X_train.T 
            grad = 2*X_train.T@(y_predict.T-y_encode)/N #DxN @ Nx10
            self.w = self.w-(grad.T + self.weight_decay*self.w)*self.lr

        return self.w

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        y_predict = self.w@X_test.T
        ans = np.argsort(y_predict,axis=0)[-1,:]
        return ans
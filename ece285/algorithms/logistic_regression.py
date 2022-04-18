"""
Logistic regression model
"""

import numpy as np

class Logistic(object):
    def __init__(self, n_class: int, lr: float, epochs: int, weight_decay: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.threshold = 0.5  # To threshold the sigmoid
        self.weight_decay = weight_decay

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # TODO: implement me
        return 1/(1+np.exp(-z))

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
        #one-hot
        y_class = np.zeros((N,self.n_class)) #Nx10
        for i in range(self.n_class):
            y_class[:,i] = 2*np.array(y_train==i,dtype=int)-1

        #gradient decent function
        for _ in range(self.epochs):
            sig = self.sigmoid(-y_class.T*(self.w@X_train.T)) #10xN
            grad = -sig*y_class.T@ X_train/N  #1xD
            self.w = self.w-(grad + self.weight_decay*self.w)*self.lr

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
        y_predict = self.sigmoid(self.w@X_test.T)
        ans = np.argsort(y_predict,axis=0)[-1,:]
        return ans

"""
K Nearest Neighbours Model
"""
import numpy as np

class KNN(object):
    def __init__(self, num_class: int):
        self.num_class = num_class

    def train(self, x_train: np.ndarray, y_train: np.ndarray, k: int):
        """
        Train KNN Classifier

        KNN only need to remember training set during training

        Parameters:
            x_train: Training samples ; np.ndarray with shape (N, D)
            y_train: Training labels  ; snp.ndarray with shape (N,)
        """
        self._x_train = x_train
        self._y_train = y_train
        self.k = k

    def predict(self, x_test: np.ndarray, k: int = None, loop_count: int = 1):
        """
        Use the contained training set to predict labels for test samples

        Parameters:
            x_test    : Test samples                                     ; np.ndarray with shape (N, D)
            k         : k to overwrite the one specificed during training; int
            loop_count: parameter to choose different knn implementation ; int

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # Fill this function in
        k_test = k if k is not None else self.k

        if loop_count == 1:
            distance = self.calc_dis_one_loop(x_test)
        elif loop_count == 2:
            distance = self.calc_dis_two_loop(x_test)

        # TODO: implement me
        nearest_k_pos = distance.argsort(axis = 1) #transform all the pos into n*k X 1 vector
        nearest_k_pos = nearest_k_pos[:,:k_test].reshape(-1)
        nearest_k_label = self._y_train[nearest_k_pos].reshape((-1,k_test)) #transform back

        #calculate mode
        ans = np.zeros(x_test.shape[0])
        k=0
        for rows in nearest_k_label:
            vals,counts = np.unique(rows, return_counts=True)
            index = np.argmax(counts)
            ans[k] = int(vals[index])
            k+=1
        
        return ans

    def calc_dis_one_loop(self, x_test: np.ndarray):
        """
        Calculate distance between training samples and test samples

        This function could contain one for loop

        Parameters:
            x_test: Test samples; np.ndarray with shape (N, D)
        """

        # TODO: implement me
        ans = np.zeros((x_test.shape[0],self._x_train.shape[0]))
        for i in range(x_test.shape[0]):
            x = x_test[i,:]
            dist = np.sum(np.square(self._x_train - x),axis=1) #(N*D - 1*D) * (N*D - 1*D) => N*D ,sum(axis = 1)
            ans[i,:] = dist
        return ans

    def calc_dis_two_loop(self, x_test: np.ndarray):
        """
        Calculate distance between training samples and test samples

        This function could contain two loop

        Parameters:
            x_test: Test samples; np.ndarray with shape (N, D)
        """
        # TODO: implement me
        ans = np.zeros((x_test.shape[0],self._x_train.shape[0]))
        for i in range(self._x_train.shape[0]):
            for j in range(x_test.shape[0]):
                tra = self._x_train[i,:]
                tes = x_test[j,:]
                dist = sum(np.square(tra-tes))
                ans[j,i] = dist
        
        return ans

    
import numpy as np


class Regression(object):

    def __init__(self):
        pass

    def rmse(self, pred, label):  # [5pts]
        '''
        This is the root mean square error.
        Args:
            pred: numpy array of length N * 1, the prediction of labels
            label: numpy array of length N * 1, the ground truth of labels
        Return:
            a float value
        '''
        diff = (pred - label) ** 2
        return np.sqrt(np.mean(diff))

    def construct_polynomial_feats(self, x, degree):  # [5pts]
        """
        Args:
            x: numpy array of length N, the 1-D observations
            degree: the max polynomial degree
        Return:
            feat: numpy array of shape Nx(degree+1), remember to include
            the bias term. feat is in the format of:
            [[1.0, x1, x1^2, x1^3, ....,],
             [1.0, x2, x2^2, x2^3, ....,],
             ......
            ]
        """
        vandermonde = np.zeros((x.shape[0], degree + 1))
        vandermonde[:, 0] = 1
        for i in range(degree):
            vandermonde[:, i + 1] = x * vandermonde[:, i]
        return vandermonde

    def predict(self, xtest, weight):  # [5pts]
        """
        Args:
            xtest: NxD numpy array, where N is number
                   of instances and D is the dimensionality of each
                   instance
            weight: Dx1 numpy array, the weights of linear regression model
        Return:
            prediction: Nx1 numpy array, the predicted labels
        """
        return xtest @ weight

    # =================
    # LINEAR REGRESSION
    # Hints: in the fit function, use close form solution of the linear regression to get weights.
    # For inverse, you can use numpy linear algebra function
    # For the predict, you need to use linear combination of data points and their weights (y = theta0*1+theta1*X1+...)

    def linear_fit_closed(self, xtrain, ytrain):  # [5pts]
        """
        Args:
            xtrain: N x D numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: N x 1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        return np.linalg.pinv(xtrain) @ ytrain

    def linear_fit_GD(self, xtrain, ytrain, epochs=5, learning_rate=0.001):  # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: Nx1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        weight = np.zeros((xtrain.shape[1], 1)) # D x 1
        for _ in range(epochs):
            weight += learning_rate * xtrain.T @ (ytrain - (xtrain @ weight)) / xtrain.shape[0]
        return weight

    def linear_fit_SGD(self, xtrain, ytrain, epochs=100, learning_rate=0.001):  # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: Nx1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        weight = np.zeros((xtrain.shape[1], 1)) # D x 1
        for i in range(epochs):
            weight += np.expand_dims(learning_rate * xtrain[i % xtrain.shape[0], :] * (ytrain[i % xtrain.shape[0]] - (xtrain[i % xtrain.shape[0], :] @ weight)), 1)
        return weight

    # =================
    # RIDGE REGRESSION

    def ridge_fit_closed(self, xtrain, ytrain, c_lambda):  # [5pts]
        """
        Args:
            xtrain: N x D numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: N x 1 numpy array, the true labels
            c_lambda: floating number
        Return:
            weight: Dx1 numpy array, the weights of ridge regression model
        """
        return np.linalg.inv(xtrain.T @ xtrain + c_lambda * np.eye(xtrain.shape[1])) @ xtrain.T @ ytrain

    def ridge_fit_GD(self, xtrain, ytrain, c_lambda, epochs=500, learning_rate=1e-7):  # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: Nx1 numpy array, the true labels
            c_lambda: floating number
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        weight = np.zeros((xtrain.shape[1], 1)) # D x 1
        for _ in range(epochs):
            weight += learning_rate * (xtrain.T @ (ytrain - (xtrain @ weight)) - 2 * c_lambda * weight) / xtrain.shape[0]
        return weight

    def ridge_fit_SGD(self, xtrain, ytrain, c_lambda, epochs=100, learning_rate=0.001):  # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: Nx1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        weight = np.zeros((xtrain.shape[1], 1)) # D x 1
        for i in range(epochs):
            weight += learning_rate *  ((np.expand_dims(xtrain[i % xtrain.shape[0], :] * (ytrain[i  % xtrain.shape[0]] - (xtrain[i  % xtrain.shape[0], :] @ weight)), 1)) - (2 * c_lambda * weight / xtrain.shape[0]))
        return weight

    def ridge_cross_validation(self, X, y, kfold=10, c_lambda=100):  # [8 pts]
        """
        Args: 
            X : NxD numpy array, where N is the number of instances and D is the dimensionality of each instance
            y : Nx1 numpy array, true labels
            kfold: Number of folds you should take while implementing cross validation.
            c_lambda: Value of regularization constant
        Returns:
            meanErrors: Float average rmse error
        Hint: np.concatenate might be helpful.
        Look at 3.5 to see how this function is being used.
        # For cross validation, use 10-fold method and only use it for your training data (you already have the train_indices to get training data).
        # For the training data, split them in 10 folds which means that use 10 percent of training data for test and 90 percent for training.
        """
        trainx = [[] for k in range(kfold)]
        testx = [[] for k in range(kfold)]
        trainy = [[] for k in range(kfold)]
        testy = [[] for k in range(kfold)]
        for i in range(X.shape[0]):
            for k in range(kfold):
                if i % kfold != k:
                    trainx[k].append(X[i])
                    trainy[k].append(y[i])
                else:
                    testx[k].append(X[i])
                    testy[k].append(y[i])
        trainx = [np.stack(d, 0) for d in trainx]
        testx = [np.stack(d, 0) for d in testx]
        trainy = [np.stack(d, 0) for d in trainy]
        testy = [np.stack(d, 0) for d in testy]
        error = np.zeros((kfold))
        for k in range(kfold):
            weight = self.ridge_fit_closed(trainx[k], trainy[k], c_lambda)
            error[k] = self.rmse(self.predict(testx[k], weight), testy[k])
        return np.mean(error)
                
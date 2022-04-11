import numpy as np
from tqdm import tqdm
    
'''
References: 
https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html#binary-logistic-regression
https://towardsdatascience.com/logistic-regression-from-scratch-in-python-ec66603592e2
https://towardsdatascience.com/implement-logistic-regression-with-l2-regularization-from-scratch-in-python-20bd4ee88a59

'''

class LogisticRegressionClassifierFromScratch(object):
    def __init__(self, epochs=100, batch_size=1,\
                        learning_rate=1e-3, l1_penalty=0):

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.l1_penalty = l1_penalty

    def sigmoid(self, z):
        return 1 / (1+np.exp(-z))

    def loss(self, y_true, y_pred):
        
        num_samples = len(y_true)
        #Take the error when label=1
        class1_loss = -y_true*np.log(y_pred)

        #Take the error when label=0
        class2_loss = (1-y_true)*np.log(1-y_pred)

        #Take the sum of both losses
        loss = class1_loss - class2_loss

        #Take the average loss
        loss = loss.sum() / num_samples

        ## Loss in one line of code
        # loss = -np.mean(y_true*(np.log(y_pred)) - (1-y_true)*np.log(1-y_pred))

        return loss

    def gradients(self, X, y_true, y_pred):
            
        '''
        X       : data (num_samples x num_features)
        y_true  : true target
        y_pred  : predicted target
        w       : weights (model parameters to be learned)
        b       : bias (model parameter to be learned)

        dw      : gradient of loss w.r.t. weights
        db      : gradient of loss w.r.t. bias
        '''
        
        num_samples = X.shape[0]
        dw = (1/num_samples)*np.dot(X.T, (y_pred-y_true))
        db = (1/num_samples)*np.sum((y_pred-y_true))

        return dw, db

            
    def train(self, X, y):

        '''
        X   : train data
        y   : train labels
        w   : weights (model parameters to be learned)
        b   : bias (model parameter to be learned)
        '''
        
        num_samples, num_features = X.shape
        
        # Initializing weights and bias to zeros.
        w = np.zeros((num_features,1))
        b = 0
        
        # Reshaping y.
        y = y.reshape(num_samples,1)
        
        # Empty list to store loss per epoch
        losses = []
        
        # Training loop.
        for epoch in tqdm(range(self.epochs)):
            for i in range((num_samples-1)//self.batch_size + 1):
                
                # Defining batches. SGD.
                start_i = i*self.batch_size
                end_i = start_i + self.batch_size
                xb = X[start_i:end_i]
                yb = y[start_i:end_i]
                
                # Calculating prediction.
                y_hat = self.sigmoid(np.dot(xb, w) + b)
                
                # Getting the gradients of loss w.r.t parameters.
                dw, db = self.gradients(xb, yb, y_hat)
                
                # Updating the parameters.
                w -= self.learning_rate*(dw + self.l1_penalty*np.sum(w))
                b -= self.learning_rate*db
            
            # Calculating loss and appending it in the list.
            l = self.loss(y, self.sigmoid(np.dot(X, w) + b))
            losses.append(l)

            # print("Epoch: {} \t loss = {}".format(epoch, l))
            
        model_params = [w,b]
        # returning weights, bias and losses.
        return model_params, losses
    
    def predict(self, X, model_params, threshold = 0.5):

        '''
        X       : test data
        model   : [w, b]
                    w   : learned weights (vector)
                    b   : learned bias (scalar)
        '''
        w = model_params[0]
        b = model_params[1]

        # Calculating presictions/y_hat.
        preds = self.sigmoid(np.dot(X, w) + b)
        
        # Empty List to store predictions.
        pred_class = []
        # if y_pred >= 0.5 --> round up to 1
        # if y_pred < 0.5 --> round up to 0
        pred_class = [1 if i > threshold else 0 for i in preds]
        
        return np.array(pred_class)
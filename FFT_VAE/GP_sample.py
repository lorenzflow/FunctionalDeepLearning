# Sample from GP Prior to generate data set for training and testing

from sklearn import gaussian_process
import numpy as np

GP = gaussian_process.GaussianProcessRegressor(optimizer='fmin_l_bfgs_b', 
                              n_restarts_optimizer=0, normalize_y=False, copy_X_train=True, random_state=None)

def sample_batch(batch_size, num_points, GP, lower, upper, x=None):
    '''Inputs: batch_size - the number of streams to sample
                num_points - points per stream
                GP - sklearn.gaussian_process.GaussianProcessRegressor object to sample from prior
                lower - lower bound for x-values
                upper - upper bound for x-values
                x (default None) - default sample X~Uniform(lower,upper), specify otherwise to sample y at equally spaced grid 
        Outputs: If x=None returns the tuple (X,Y)
                    X - np array of locations with dimension (batch_size, num_points, 1)
                    Y - np array of prior samples with dimensions (batch_size, num_points, 1)
                If x!= None the tuple (x,Y)
                    x - equally spaced grid between lower and upper (num_points,)
                    Y - np array of prior samples with dims (num_points, batch_size, 1)'''
        
    if x==None:
        X = np.random.uniform(lower, upper, (batch_size,num_points))
        Y = np.zeros_like(X)
        for i in range(batch_size):
            Y[i,:] = np.squeeze(GP.sample_y(X[i,:].reshape(-1, 1)))
        return np.expand_dims(X,-1),np.expand_dims(Y,-1)
    else: # num_ points equally spaced points between lower and upper
        x = np.linspace(lower, upper, num_points)
        Y = GP.sample_y(x[:, np.newaxis], batch_size)
        return x, np.expand_dims(Y,-1)
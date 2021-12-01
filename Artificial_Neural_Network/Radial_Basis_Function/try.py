import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class MLP:
    def __init__(self, in_dim, hid_dim, out_dim):
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.W, self.V = self.init_weights()

    # extra col for bias
    #Initialize W and V
    def init_weights(self):
        return (
            np.random.normal(0,1e-3, size=(self.hid_dim,self.in_dim+1)),
            np.random.normal(0,1e-3, size=(self.out_dim,self.hid_dim+1))
        )
    def forward(self, X):
        #print(self.W.shape)
        Hin = self.W @ X
        #print(Hin.shape)
        _, num_samples = Hin.shape
        #Hin = np.concatenate((Hin, np.ones((1,num_samples))), axis=0)
        #dsig1 = dsig(Hin)
        #Hin = np.concatenate((Hout, np.ones((1,num_samples))), axis=0)
        Hout = sig(Hin)        
        #Hout_1 = np.concatenate((Hout, np.ones((1,num_samples))), axis=0)
        Hin = np.concatenate((Hin, np.ones((1,num_samples))), axis=0)
        dsig1 = dsig(Hin)
        #Hout_bias = np.concatenate((Hout, np.ones((1,num_samples))), axis=0)#will be removed later in train

        Hout = np.concatenate((Hout, np.ones((1,num_samples))), axis=0)
        # add ow for bias
        ##here#########################333
        #print(Hout_bias.shape)
        #print(self.V.shape)
        #self.V[0,-1]=1
        #print(self.V)
        Oin = self.V @ Hout
        #Oin = self.V[:,:-1] @ Hout
        # Oout = sig(Oin)
        # dsig2 = dsig(Oin)
        # Difference for function approx.
        Oout = sig(Oin)
        dsig2 = dsig(Oin)
        return Hout, dsig1, Oout, dsig2

    def train(self, X, labels, learning_rate=0.01):
        X = X.reshape((X.shape[0], 1)).T
        X = np.concatenate((X, np.ones(X.shape)), axis=0)
        T = labels.reshape((labels.shape[0], 1)).T
        eta = learning_rate
        epochs = 500
        a = 0.9

        _, N = X.shape
        msg_epoch = []

        # TODO: is this assumption correct?
        theta = 0.
        psi = 0.
        msg = 10000 

        # initial forward
        H, dsig_hid, O, dsig_out = self.forward(X)
        for epoch in range(epochs):
            # TODO: fix sequential
            
            # backward (forward is already done)
            Odelta = np.multiply((O - T), dsig_out)
            Hdelta = np.multiply((self.V.T @ Odelta), dsig_hid)
            Hdelta = np.delete(Hdelta,-1,0)
            print("Hdelta:",Hdelta.shape)
            theta = a*theta - (1-a)*Hdelta @ X.T
            #print("#######################",Odelta.shape,H.T.shape)
            psi = a*psi - (1-a)*Odelta @ H.T
            #print(psi)
            print("every epoch, theta:",theta)
            print("every epoch, psi:",psi)
            self.W += eta*theta
            ##here###########################
            #print(self.V.shape)
            self.V += eta*psi
            #self.V[:,:-1] += eta*psi

            '''
            if np.sum(np.abs(old_W - self.W)) < thres:
                print('converged at epoch: {}'.format(epoch))
                return epoch, msg_epoch
            '''
            
            # log mean square error in training set
            e = (T - O)
            old_msg = msg
            msg = e @ e.T * 1/N
            msg = msg[0,0]
            msg_epoch.append(msg)

        return msg_epoch
    
    def test(self, X, labels):
        X = X.reshape((X.shape[0], 1)).T
        X = np.concatenate((np.ones(X.shape), X), axis=0)
        T = labels.reshape((labels.shape[0], 1)).T

        _, N = X.shape
        H, dsig_hid, prediction, dsig_out = self.forward(X)
        print("In test:",H)
        e = (T - prediction)
        mse = e @ e.T * 1/N
        return mse, prediction



# activation functions
def sig(x):
    return 2. / (1. + np.exp(-x)) - 1.
def dsig(x):
    return (1. + sig(x))*(1. - sig(x)) / 2.
def sig22(x):
    return x
def dsig22(x):
    return 1

class RBFNETWORK:
    def __init__(self, X, cl_lr, units, data_space=[0, (2*np.pi)], competative_learning=True):
        # use uniform distributed means for RBF nodes in the data space
        step_size=((data_space[0] + data_space[1]) /units)
        if len(X.shape) > 1:
            if competative_learning:
                #self.node_means = np.array( [data_space[1]*np.random.rand(units) for m in range(X.shape[1]) ] ).T
                self.node_means = []
                for u in range(units):
                    rand_idx = np.random.choice(len(X))
                    rand_point = X[rand_idx]
                    self.node_means.append(rand_point)
                self.node_means = np.array(self.node_means)
                
                self.initial_means = self.node_means.copy()
                self.dim = len(self.node_means)
                # self.cl(X, cl_lr)
                self.concious_levels = np.ones(self.dim)
                self.comp_mean_learning(X, eta=cl_lr, cl_epchs=20)
            else: 
                self.node_means = np.array( [list(zip(np.arange(data_space[0], data_space[1], step_size*2), np.zeros(int(units/2))+m)) for m in [0.25,0.75] ] )
                self.node_means = self.node_means.reshape((20,2))
                self.dim = len(self.node_means)
            self.std = 0.85
            means = np.ones(X.shape[1]) * 0.0
            cov = np.zeros((X.shape[1],X.shape[1]), float)
            np.fill_diagonal(cov, 0.000001)
            self.weights = np.random.multivariate_normal(means, cov, units)
        else:
            if competative_learning:
                self.node_means = data_space[1]*np.random.rand(units)
                self.node_means = self.node_means.reshape((units, 1))
                self.dim = len(self.node_means)
                #self.cl(X, cl_lr)
                self.concious_levels = np.ones(self.dim)
                self.comp_mean_learning(X, eta=cl_lr, cl_epchs=20)
            else: 
                self.node_means = np.arange(data_space[0], data_space[1], step_size)
                self.node_means = self.node_means.reshape((units, 1))
                self.dim = len(self.node_means)
            self.std = 0.4
            self.weights = np.random.normal(0, 0.1, self.dim).T
        # all nodes have same variance/std.

    def cl(self, X, learning_rate=0.01, cl_epchs=10, num_neighbor_updates=5):
        if len(X.shape) > 1:
            X = X.reshape((len(X), len(X[0])))
            data_dim = 2
        else:
            X = X.reshape((len(X), 1))
            data_dim = 1
        reshape_means = self.node_means #.reshape(self.dim, data_dim)

        for epoch in range(cl_epchs): # epochs
            if epoch > 0:
                num_neighbor_updates = int((num_neighbor_updates+1)*(1-epoch/cl_epchs))
            for data_idx in range(int(len(X))):
                dist = np.linalg.norm(X[data_idx] - reshape_means, axis=1)
                winner_idx = np.argmin(dist)
                dist_between_means = np.linalg.norm(self.node_means[winner_idx] - reshape_means, axis=1)

                o = 0.05
                neighborhood = np.exp(-dist_between_means**2/(2*( o * np.exp(-epoch/cl_epchs ))**2))
                neighborhood = neighborhood.reshape((self.dim,1))
                x = X[data_idx]
                dist_winner_data = x-self.node_means[winner_idx]
                self.node_means[winner_idx] += learning_rate * (dist_winner_data)
                neighbor_idx = np.where(dist[winner_idx] < dist)
                self.node_means[neighbor_idx] += learning_rate * neighborhood[neighbor_idx] * (x-self.node_means[neighbor_idx])
    
    def comp_mean_learning(self, data, eta=0.01, cl_epchs=20): # with concious level
        for epoch in range(cl_epchs):
            for x in data:
                dist = np.linalg.norm(x - self.node_means, axis=1)
                scaled_dist = dist * self.concious_levels
                winner_idx = np.argmin(scaled_dist)
                self.node_means[winner_idx]  += eta * (x-self.node_means[winner_idx])
                # update conciousness levels
                for level in range(len(self.concious_levels)):
                    if level == winner_idx:
                        self.concious_levels[winner_idx] = 1.0
                    else:
                        self.concious_levels[winner_idx] *= 0.9
            
    def radialBaseFunction(self, x, mu, sigma):
        if type(x-mu) != type(np.array([])):
            norm = np.linalg.norm(np.array([x-mu]), 2) 
            #return (np.exp(-(norm)/(2*sigma**2))) / np.sum((np.exp(-(np.linalg.norm([x-self.node_means], 2))/(2*sigma**2)))) # normalize
            return np.exp(-(norm**2)/(2*sigma**2))
        else:
            norm = np.linalg.norm(x-mu, 2) 
            return np.exp(-(norm**2)/(2*sigma**2))

    def batchForward(self, X):
        N = X.shape[0]
        PHI = np.empty([N,self.dim])
        for x_idx in range(N):
            for node_idx in range(self.dim):
                x = X[x_idx]
                phi_el = self.radialBaseFunction(x, self.node_means[node_idx], self.std)
                PHI[x_idx, node_idx] = phi_el
        prediction = PHI @ self.weights
        return PHI, prediction

    def sequentialForward(self, x):
        """
        PHI = np.empty([self.dim]) # row vect
        for node_idx in range(self.dim):
            PHI[node_idx] = self.radialBaseFunction(x, node_idx)
        ret = PHI.T.dot(self.weights) # ret needs to be scalar
        """
        phi_x = np.array([self.radialBaseFunction(x, m, self.std) for m in self.node_means])
        pred = phi_x.T @ self.weights
        return phi_x, pred

    def train_lst(self, patterns, labels):
            PHI, _ = self.batchForward(patterns)
            self.weights += np.linalg.lstsq(PHI,labels,rcond=None)[0]

    def train_sequential(self, patterns, labels, learningRate, epochs):
        for epoch in range(epochs):
            patterns, labels = shuffleData(patterns, labels)
            for point_idx in range(len(patterns)):
                PHI_col, prediction = self.sequentialForward(patterns[point_idx])
                delta = learningRate * (labels[point_idx] - prediction) * PHI_col
                self.weights += delta

    def test(self, test_patterns, test_labels, thresholding=False):
        N = test_patterns.shape[0]
        labels = test_labels.copy()
        _, prediction = self.batchForward(test_patterns)
        mse = np.linalg.norm(labels - prediction)
        if thresholding:
            out_transform = np.piecewise(prediction, [prediction >= 0, prediction < 0], [1,-1])
            e_transform = np.linalg.norm(out_transform - labels)
            mse_trans = e_transform*e_transform
            return mse, prediction, mse_trans, out_transform
        return mse, prediction 




        
def shuffleData(patterns, labels):
    indexes = np.random.permutation(len(patterns))
    X_shuff = patterns[indexes]
    T_shuff = labels[indexes]
    return X_shuff, T_shuff

def sin(x):
    return np.sin(x)
def square(x):
    return np.piecewise(x, [sin(x) >= 0, sin(x) < 0], [1,-1])

def generateData(noise=True): # TODO change noise to be on labels
    X_train = np.arange(0,2*np.pi, 0.1)
    N_train = X_train.shape[0]
    X_test = np.arange(0.05,2*np.pi, 0.1)
    N_test = X_test.shape[0]
    if noise:
        training_noise = np.random.normal(0,np.sqrt(0.1), N_train)
        X_train += training_noise
        test_noise = np.random.normal(0,np.sqrt(0.1), N_test)
        X_test += test_noise
    labels_sin_train = sin(2*X_train)
    labels_square_train = square(2*X_train)
    labels_sin_test = sin(2*X_test)
    labels_square_test = square(2*X_test)
    return X_train, X_test, labels_sin_train, labels_sin_test, labels_square_train, labels_square_test

def experiment1():
    epochs = 200
    learningRate = 0.01
    CL_learningRate = 0.01
    X_train, X_test, labels_sin_train, labels_sin_test, labels_square_train, labels_square_test = generateData()
    # create model
    rbf_batch_sin = RBFNETWORK(X_train, CL_learningRate, 20)
    rbf_sequential_sin = RBFNETWORK(X_train, CL_learningRate, 20)
    rbf_batch_square = RBFNETWORK(X_train, CL_learningRate, 20)
    rbf_sequential_square = RBFNETWORK(X_train, CL_learningRate, 20)
    # training
    rbf_batch_sin.train_lst(X_train, labels_sin_train)
    rbf_sequential_sin.train_sequential(X_train, labels_sin_train, learningRate, epochs)
    rbf_batch_square.train_lst(X_train, labels_square_train)
    rbf_sequential_square.train_sequential(X_train, labels_square_train, learningRate, epochs)
    # testing
    rbf_batch_sin_error, rbf_batch_sin_prediction = rbf_batch_sin.test(X_test, labels_sin_test)
    rbf_sequential_sin_error, rbf_sequential_sin_prediction = rbf_sequential_sin.test(X_test, labels_sin_test)
    rbf_batch_square_error, rbf_batch_square_prediction, rbf_batch_square_smoothed_mse, rbf_batch_square_smoothed_prediction = rbf_batch_square.test(X_test, labels_square_test, thresholding=True)
    rbf_sequential_square_error, rbf_sequential_square_prediction, rbf_sequential_square_smoothed_mse, rbf_sequential_square_smoothed_prediction = rbf_sequential_square.test(X_test, labels_square_test, thresholding=True)
    
    # ploting for sinus wave
    fig, (ax1,ax2,ax3,ax4) = plt.subplots(4)
    ax1.plot(X_test, labels_sin_test, 'g', label='ground truth sinus')
    ax1.plot(X_test, rbf_batch_sin_prediction, 'r', alpha=0.7, label='batch sinus')
    ax1.plot(rbf_batch_sin.node_means, np.zeros(rbf_batch_sin.dim), "b*")
    ax1.legend(loc='upper left')

    ax2.plot(X_test, labels_sin_test, 'g', label='ground truth sinus')
    ax2.plot(X_test, rbf_sequential_sin_prediction, 'r', alpha=0.7, label='sequential sinus')
    ax2.plot(rbf_sequential_sin.node_means, np.zeros(rbf_sequential_sin.dim), "b*")
    ax2.legend(loc='upper left')

    ax3.plot(X_test, labels_square_test, 'g', label='ground truth square')
    ax3.plot(X_test, rbf_batch_square_smoothed_prediction, 'r', alpha=0.7, label='batch square')
    ax3.plot(rbf_batch_square.node_means, np.zeros(rbf_batch_square.dim), "b*")
    ax3.legend(loc='upper left')

    ax4.plot(X_test, labels_square_test, 'g', label='ground truth square')
    ax4.plot(X_test, rbf_sequential_square_smoothed_prediction, 'r', alpha=0.7, label='sequential square')
    ax4.plot(rbf_sequential_square.node_means, np.zeros(rbf_sequential_square.dim), "b*")
    ax4.legend(loc='upper left')
    plt.show()

def balistic2d():
    # conf
    comp_learn_rate = 0.01
    
    # get data
    """
    train_data = pd.read_csv('./data_lab2/ballist.dat', sep=' |\t', names=['angle','velocity','distance','height'], engine='python')
    test_data = pd.read_csv('./data_lab2/balltest.dat', sep=' |\t', names=['angle','velocity','distance','height'], engine='python')
    """
    train_data = pd.read_csv('/Users/pacmac/iCloud Drive (Archive)/Documents/Uni/KTH/DD2437-ANN/neural-nets/lab2/data_lab2/ballist.dat', sep=' |\t', names=['angle','velocity','distance','height'], engine='python')
    test_data = pd.read_csv('/Users/pacmac/iCloud Drive (Archive)/Documents/Uni/KTH/DD2437-ANN/neural-nets/lab2/data_lab2/balltest.dat', sep=' |\t', names=['angle','velocity','distance','height'], engine='python')
    
    train_X = train_data[['angle','velocity']].to_numpy()
    train_T = train_data[['distance','height']].to_numpy()
    test_X = test_data[['angle','velocity']].to_numpy()
    test_T = test_data[['distance','height']].to_numpy()
    # model
    fig, (ax1,ax2) = plt.subplots(2)
    rbf_ballist = RBFNETWORK(train_X, comp_learn_rate, units=20, data_space=[0,1])
    rbf_ballist.train_lst(train_X, train_T)
    error_ballist, prediction_ballist = rbf_ballist.test(test_X,test_T)
    # plot
    print('Ploting...')
    
    ax1.plot(train_X.T[0], train_X.T[1], 'y*')
    #ax1.plot(test_X.T[0], test_X.T[1], 'b*')
    #ax1.plot(rbf_ballist.initial_means.T[0], rbf_ballist.initial_means.T[1], 'r+', alpha=1.0)
    ax1.plot(rbf_ballist.node_means.T[0], rbf_ballist.node_means.T[1], 'r+', alpha=1.0)
    ax1.set_title('Distribution of RBF nodes in test input space')
    
    ax2.plot(test_T.T[0], test_T.T[1], 'g*')
    #ax2.plot(train_T.T[0], train_T.T[1], 'y+')
    #ax2.plot( np.abs( prediction_ballist.T[0] ), np.abs( prediction_ballist.T[1] ), 'r+')
    ax2.plot(prediction_ballist.T[0], prediction_ballist.T[1], 'r+')
    ax2.set_title('Prediction in the outputspace')
    
    print('Test error is: ', error_ballist)
    plt.show()

def experiment2(): #MLP
    X_train, X_test, labels_sin_train, labels_sin_test, labels_square_train, labels_square_test = generateData()
    # config
    in_dim, hid_dim, out_dim = 1, 20,  1
    eta = 0.01
    mlp_sin = MLP(in_dim, hid_dim, out_dim)
    mlp_square = MLP(in_dim, hid_dim, out_dim)

    sin_train_mses = mlp_sin.train(X_train, labels_sin_train)
    sin_test_mse, sin_prediction = mlp_sin.test(X_test, labels_sin_test)
    square_train_mses = mlp_square.train(X_train, labels_square_train)
    square_test_mse, square_prediction = mlp_square.test(X_test, labels_square_test)
    plt.plot(sin_train_mses,'g') 
    print(sin_prediction)
    print(square_prediction)

    # plot
    fig, (ax1,ax2) = plt.subplots(2)
    ax1.plot(X_test, labels_sin_test, 'g', label='ground truth sinus')
    ax1.plot(X_test, sin_prediction.reshape((len(X_test))), 'r', alpha=0.7, label='MLP sinus')
    ax1.legend(loc='upper left')
    ax2.plot(X_test, labels_square_test, 'g', label='ground truth squared')
    ax2.plot(X_test, square_prediction.reshape((len(X_test))), 'r', alpha=0.7, label='MLP squared')
    ax2.legend(loc='upper left')
    plt.show()

#experiment1()
experiment2()
#balistic2d()

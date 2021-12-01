import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Net1:
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W = self.init_weights()

    # initialises with std gaussian noise
    # extra rows for bias
    def init_weights(self):
        return np.random.normal(0, 1e-4, 
            size=(self.out_dim,self.in_dim+1)) # +1 for bias
    
    def reset_weights(self):
        self.W = self.init_weights()

    def forward_batch(self, X):
        return self.W @ X

    def forward(self, x):
        return self.W.dot(x)

    def delta_update_batch(self,W,X,T,eta):
        #print(X.shape)
        return W - eta*(W @ X - T) @ X.T

    def train(self, config):
        X = config['data']
        T = config['labels']
        epochs = config['epochs']
        update_type = config['update_type']
        thres = config['threshold']
        eta = config['learn_rate']
        _, N = X.shape

        for epoch in range(epochs):
            if update_type == 'batch':
                W_old = self.W
                self.W += self.delta_update_batch(W_old,X,T,eta)
                print(self.W)
                plot_data(config, self, -6, 6, convergence_epoch=1)
            elif update_type == 'sequential':
                for i in range(N):
                    self.W = delta_update_seq(self.W,X[:,i],T[:,i],eta) 
                plot_data(config, self, -6, 6, convergence_epoch=1)
            elif update_type == 'perceptron':
                for i in range(N):
                    self.W, changed = perceptron_update(self.W,X[:,i],T[:,i],eta)
                plot_data(config, self, -6, 6, convergence_epoch=1)
            else:
                raise Exception('Unknown update option')
    
    def get_boundary(self,x):
        return -(self.W[0,0]/self.W[0,1]) * x - (self.W[0,2]/self.W[0,1])

# batch X,T inputs and targets each column
# W = {w_ji} weight from x_i to t_j
# adds the negative gradient to minimize error

def delta_update_batch(W,X,T,eta):
    #print(X.shape)
    return -eta*(W @ X - T) @ X.T 

def delta_update_seq(W,x,t,eta):
    new_W = W - eta*(W.dot(x) - t) * x.T
    print(x.shape)
    return new_W

def perceptron_update(W, x, t, eta):
    if W.dot(x) > 0 and t == -1: # y = 1, t = -1
        new_W = W - eta*x.T, True
        print(new_W)
        return new_W
    if W.dot(x) <= 0 and t == 1: # y = -1, t = 1
        new_W = W + eta*x.T, True
        print(new_W)
        return new_W
    # returns the weights and if they got changed
    return W, False

def sig(x):
    return 2. / (1. + np.exp(-x)) - 1.

def dsig(x):
    return (1. + sig(x))*(1. - sig(x)) / 2.

def data_gen(N, m_a, m_b, sig_a, sig_b, x_dim=2, out_dim=1):
    class_a = np.random.multivariate_normal(m_a, sig_a, N)
    class_b = np.random.multivariate_normal(m_b, sig_b, N)
    T_a = np.ones((out_dim,N))
    T_b = -1*np.ones((out_dim,N))
    T = np.concatenate((T_a,T_b), axis=1)
    data = np.concatenate((class_a, class_b), axis=0).T
    return data, T

def plot_data(config, network, line_start, line_end, convergence_epoch=1):
    data = np.delete(config['data'], (-1), axis=0)
    x, y = data
    plt.scatter(x, y, c=config['labels'], cmap='viridis')
    plt.axis('equal')
    
    # calculate bouandary 
    x_cord = np.linspace(line_start,line_end)
    y_cord = network.get_boundary(x_cord)
    plt.plot(x_cord, y_cord, 'k-')
    plt.xlim([line_start, line_end])
    plt.ylim([line_start, line_end])

    plt.colorbar()
    plt.savefig(
        'plots/{}_update_lr_{}_{}_epochs.png'.format(
            config['update_type'], config['learn_rate'], convergence_epoch
         ))
    plt.show()
    plt.clf()

def plot_mse(config, mses, mse_conv_epoch):
    if len(mses) == 0:
        #print(mses[0])
        plt.plot(x=0, y=mses[0]) 
    else:
        plt.plot(mses)
    plt.title('Mean square error for {} update function, with learning rate {}:'.format(config['update_type'], config['learn_rate']))
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.savefig('mse/{}_update_lr_{}_{}_epochs.png'.format(config['update_type'], config['learn_rate'], mse_conv_epoch))
    # plt.show()
    plt.clf()

def main():
    # set seed
    np.random.seed(4357578)

    # config layers
    x_dim = 2
    out_dim = 1
    N = 100

    # data generation
    mean_a = np.array([0,0])
    mean_b = np.array([4,4])
    sigma_a = np.array([[0.5,0.0], [0.0,0.5]])
    sigma_b = np.array([[0.5,0.0], [0.0,0.5]])
    data, labels = data_gen(N, mean_a, mean_b, sigma_a, sigma_b, 2, 1)
    data = np.concatenate((data, np.ones((1,2*N))), axis=0)

    # Network objects
    net1 = Net1(x_dim, out_dim)
    print("Net initialised ... ")
    

    # training config
    config = {
        'data': data,
        'labels': labels,
        'epochs': 100,
        'learn_rate': 0.001,
        'update_type': 'batch',
        'threshold': 1e-5,
        'alpha': 0.9
    }

    #print(config['data'].shape)
    #print(config['data'])
    #return 0

    # plot all
    """
    batch_lr = [0.001]#[0.0015, 0.001, 0.0005]#[0.002, 0.001, 0.0009]
    for ut in ['batch', 'perceptron', 'sequential']:
        config['update_type'] = ut
        for lr in batch_lr:
            net1.reset_weights()
            convergence_epoch, mse_epoch = net1.train(config)
            config['learn_rate'] = lr
            # create plots
            plot_data(config, net1, -8, 8, convergence_epoch, mse_epoch)
            plot_mse(config, mse_epoch, convergence_epoch)
    """

    # plot single 
    net1.train(config)
    print("Training finished ...")
    plot_data(config, net1, -4, 4)
    

if __name__ == "__main__":
    main()

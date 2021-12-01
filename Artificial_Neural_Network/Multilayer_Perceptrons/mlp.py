import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

bias_off = False

class Net1:
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W = self.init_weights()

    # initialises with std gaussian noise
    # extra rows for bias
    def init_weights(self):
        if bias_off:
            return np.random.normal(0, 1e-3, 
                size=(self.out_dim,self.in_dim))
        else:
            return np.random.normal(0, 1e-3, 
                size=(self.out_dim,self.in_dim+1))
    
    def reset_weights(self):
        self.W = self.init_weights()

    def forward_batch(self, X):
        return self.W @ X

    def forward(self, x):
        return self.W.dot(x)
        #return self.W @ x
    
    def delta_update_batch(self,X,T,eta):
        return -eta*(self.W @ X - T) @ X.T

    def train(self, config):
        X = config['data']
        T = config['labels']
        epochs = config['epochs']
        update_type = config['update_type']
        thres = config['threshold']
        eta = config['learn_rate']
        a = config['alpha']

        _, N = X.shape
        mse_epoch = []
        class_errs = []

        for epoch in range(epochs):
            if update_type ==  'batch':
                old_W = self.W.copy()
                # print per epoch
                if False:
                    plot_data(config, self, -6, 6, epoch)
                self.W += self.delta_update_batch(X,T,eta)
                if np.sum(np.abs(old_W - self.W)) < config['threshold']:
                    print('converged at epoch: {}'.format(epoch))
                    return epoch, mse_epoch, class_errs
            elif update_type == 'sequential':
                for i in range(N):
                    old_W = self.W.copy()
                    self.W = delta_update_seq(self.W,X[:,i],T[:,i],eta) 
                    if np.sum(np.abs(old_W -self.W)) < config['threshold']: 
                        print('Converged at epoch: ', epoch)
                        return epoch, mse_epoch, class_errs
            elif update_type == 'perceptron':
                converged = True
                for i in range(N):
                    self.W, changed = perceptron_update(
                        self.W,X[:,i],T[:,i],eta
                    )
                    if changed:
                        converged = False
                if converged:
                    print('Converged at epoch {}'.format(epoch))
                    return epoch, mse_epoch, class_errs
            else:
                raise Exception('Unknown update option')
            # error logging
            e = (T - self.W @ X)
            #print("\n\n--------")
            #print(e)
            mse = e @ e.T * 1/N
            mse = mse[0,0]
            # print(mse, epoch, mse.shape)

            mse_epoch.append(mse)

            O = self.W @ X
            class_err = 1 - np.sum(np.vectorize(classify)(O,T)) / N
            class_errs.append(class_err)

        print('ALl epochs run!')
        return epochs, mse_epoch, class_errs
    
    def get_boundary(self,x):
        if bias_off:
            return -(self.W[0,0]/self.W[0,1]) * x
        else:
            return -(self.W[0,0]/self.W[0,1]) * x - (self.W[0,2]/self.W[0,1])

class Net2:
    def __init__(self, in_dim, hid_dim, out_dim):
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.W, self.V = self.init_weights()

    # extra col for bias
    #Initialize W and V
    def init_weights(self):
        return (
            np.random.normal(0,1e-4, size=(self.hid_dim+1,self.in_dim+1)),
            np.random.normal(0,1e-4, size=(self.out_dim,self.hid_dim+1))
        )
    def forward(self, X):
        Hin = self.W @ X
        _, num_samples = Hin.shape
        #Hin = np.concatenate((Hin, np.ones((1,num_samples))), axis=0)
        #Hout = sig(Hin)
        #dsig1 = dsig(Hin)
        dsig1 = dsig(Hin)
        Hout = sig(Hin)
        # add ow for bias
        print("W in forward:",self.W.shape)        
        print("V in forward:",self.V.shape)
        Oin = self.V @ Hout
        Oout = sig(Oin)
        dsig2 = dsig(Oin)
        #return Hin, Hout, dsig1, Oin, Oout, dsig2
        return Hout, dsig1, Oout, dsig2

    def train(self, config):
        X = config['data']
        T = config['labels']
        update_type = config['update_type']
        thres = config['threshold']
        eta = config['learn_rate']
        a = config['alpha']

        _, N = X.shape
        msg_epoch = []

        # TODO: is this assumption correct?
        theta = 0.
        psi = 0.
        for epoch in range(config['epochs']):
            # TODO: fix sequential
            old_W = self.W.copy()
            print("bias in W:",old_W[-1][:])
            old_V = self.V.copy()
            
            #forward
            H, dsig_hid, O, dsig_out = self.forward(X)
            # backward
            Odelta = np.multiply((O - T), dsig_out) 
            #print("Odelta:",Odelta.shape)
            #print("V.T:",self.V.T.shape)
            #print("dsig:",dsig_hid.shape)
            Hdelta = np.multiply((self.V.T @ Odelta), dsig_hid)
            #print("Hdelta:",Hdelta.shape)
            #print("dsig_hid:",dsig_hid.shape)
            theta = a*theta - (1-a)*Hdelta @ X.T
            psi = a*psi - (1-a)*Odelta @ H.T
            # update
            #print(self.W.shape)
            #print("theta:",theta.shape)
            self.W += eta*theta
            self.V += eta*psi
            '''
            if np.sum(np.abs(old_W - self.W)) < thres:
                print('converged at epoch: {}'.format(epoch))
                return epoch, msg_epoch
            '''
            # log error
            e = (T - O)
            msg = e @ e.T * 1/N
            print(msg[0,0], epoch, msg.shape)
            msg_epoch.append(msg[0,0])
        
        #print('No convergence!')
        return config['epochs'], msg_epoch

# end net2 class 

def plot_msg(config, msgs):
    plt.plot(msgs)
    plt.title('Mean square error for {} update function, with learning rate {}:'.format(config['update_type'], config['learn_rate']))
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.savefig('msg/{}_update_lr_{}_{}_epochs.png'.format(config['update_type'], config['learn_rate'], config['epochs']))
    plt.show()
        
def delta_update_seq(W,x,t,eta):
    # correct operators??
    return W - eta*(W.dot(x) - t) * x.T

def perceptron_update(W, x, t, eta):
    if W.dot(x) > 0 and t == -1: # y = 1, t = -1
        return W - eta*x.T, True
    if W.dot(x) <= 0 and t == 1: # y = -1, t = 1
        return W + eta*x.T, True
    # returns the weights and if they got changed
    return W, False


# returns 1 if correctly classified, otherwise 0
def classify(o, t):
    return int((o > 0 and t == 1) or (o <= 0 and t == -1))

# batch X,T inputs and targets each column
# W = {w_ji} weight from x_i to t_j
# adds the negative gradient to minimize error

def sig(x):
    return 2. / (1. + np.exp(-x)) - 1.

def dsig(x):
    return (1. + sig(x))*(1. - sig(x)) / 2.

def data_gen(N, m_a, m_b, sig_a, sig_b, x_dim=2, out_dim=1):
    class_a = np.random.multivariate_normal(m_a, sig_a, N)
    np.random.shuffle(class_a)
    class_b = np.random.multivariate_normal(m_b, sig_b, N)
    np.random.shuffle(class_b)
    T_a = np.ones((out_dim,N))
    T_b = -1*np.ones((out_dim,N))
    T = np.concatenate((T_a,T_b), axis=1)
    data = np.concatenate((class_a, class_b), axis=0).T
    return data, T

def sub_sample(data, labels, experiment):
    # remember data is already random since its randomly sampled
    n = len(labels)
    data = np.concatenate((data, labels), axis=0)
    class_a = data[:,0:100]
    class_b = data[:,100:200]
    if experiment == 1: # 25% of each
        data = np.concatenate((class_a[:,0:75], class_b[:,0:75]), axis=1)
        labels = data[3,:]
        data = data[0:3,:]
    elif experiment == 2: # 50% removed from a
        data = np.concatenate((class_a[:,0:50], class_b[:,0:100]), axis=1)
        labels = data[3,:]
        data = data[0:3,:]
    elif experiment == 3: # 50% removed from b
        data = np.concatenate((class_a[:,0:100], class_b[:,0:50]), axis=1)
        labels = data[3,:]
        data = data[0:3,:]
    elif experiment == 4:
        # 20,80 conditional sample of class A
        sub_20 = []
        sub_80 = []
        for col in class_a.T:
            if col[1] < 0:
                sub_20.append(col)
            else:
                sub_80.append(col)
        sub_20 = np.array(sub_20).T
        sub_80 = np.array(sub_80).T
        sub_class_a = np.concatenate((sub_20[:,0:10], sub_80[:,0:40]), axis=1)
        data = np.concatenate((sub_class_a, class_b), axis=1)
        labels = data[3,:]
        data = data[0:3,:]
    else:
        raise Exception('invalide experiment configuration')
    return data, labels

def plot_data(config, network, line_start, line_end, convergence_epoch):
    # since bias add-on has to be deleted double check mode
    if not bias_off:
        data = np.delete(config['data'], (-1), axis=0)
    else:
        data = config['data']
    x, y = data
    plt.scatter(x, y, c=config['labels'], cmap='viridis')
    # plt.axis('equal')
    
    # calculate bouandary 
    x_cord = np.linspace(line_start,line_end)
    y_cord = network.get_boundary(x_cord)
    plt.plot(x_cord, y_cord, 'k-')
    plt.xlim([line_start, line_end])
    plt.ylim([line_start, line_end])

    plt.colorbar()
    plt.savefig(
        'plots/{}_update_lr_{}_epochs_{}.png'.format(
            config['update_type'], config['learn_rate'], convergence_epoch
         ))
    #plt.show()
    plt.clf()

def plot_err(config, errs, mse_conv_epoch, err_label):
    plt.plot(errs)
    plt.title('{} for {} update function, with learning rate {}:'.format(
        err_label, config['update_type'], config['learn_rate']))
    plt.xlabel('Epochs')
    plt.ylabel('Errs')
    plt.savefig('err/{}_{}_update_lr_{}_{}_epochs.png'.format(
        err_label,config['update_type'], 
        config['learn_rate'], mse_conv_epoch))
    #plt.show()
    plt.clf()

def main():
    # set seed
    np.random.seed(100)

    # config layers
    x_dim = 2
    out_dim = 1
    N = 100

    # data generation
    #linear separable
    """
    mean_a = np.array([1,0.5])
    mean_b = np.array([-4.0,0.])
    sigma_a = np.array([[0.5,0.0], [0.0,0.5]])
    sigma_b = np.array([[0.5,0.0], [0.0,0.5]])
    """
    
    # given example config, non-linear separable
    mean_a = np.array([1.0,0.3])
    mean_b = np.array([0.0,-0.1])
    sigma_a = np.array([[0.2,0.0], [0.0,0.2]])
    sigma_b = np.array([[0.3,0.0], [0.0,0.3]])
    """
    #non-linear separable
    mean_a = np.array([1.0,1.0])
    mean_b = np.array([0.4,0.4])
    sigma_a = np.array([[0.5,0.0], [0.0,0.5]])
    sigma_b = np.array([[0.5,0.0], [0.0,0.5]])
    """
    data, labels = data_gen(N, mean_a, mean_b, sigma_a, sigma_b, 2, 1)
    # add ones for bias term
    if not bias_off:
        data = np.concatenate((data, np.ones((1,2*N))), axis=0)

    #sub-sampleing class_a TODO
    data, labels = sub_sample(data, labels, 3)

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
        'threshold': 1e-7,
        'alpha': 0.9
    }

    # plot all
    """
    batch_lr = [0.001, 0.0015, 0.0009]
    for ut in ['batch', 'perceptron', 'sequential']:
        config['update_type'] = ut
        for lr in batch_lr:
            config['learn_rate'] = lr
            net1.reset_weights()
            convergence_epoch, mse_epoch, class_errs = net1.train(config)

            # create plots
            plot_data(config, net1, -6, 6, convergence_epoch)
            plot_err(
                config, mse_epoch, 
                convergence_epoch, 'mse'
            )
            plot_err(
                config, class_errs, 
                convergence_epoch, 'classification'
            )
    """
    # plot single 
    print('Used data has shape: {}'.format(data.shape))
    plot_data(config, net1, -6, 6, 0)
    convergence_epoch, mse_epoch, class_errs = net1.train(config)
    print("Training finished ...")
    plot_data(config, net1, -6, 6, convergence_epoch)
    plot_err(
        config, mse_epoch, 
        convergence_epoch, 'mse'
    )
    plot_err(
        config, class_errs, 
        convergence_epoch, 'classification'
    )

if __name__ == "__main__":
    main()

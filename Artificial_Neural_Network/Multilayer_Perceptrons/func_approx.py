import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math

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
            np.random.normal(1,1e-2, size=(self.hid_dim,self.in_dim+1)),
            np.random.normal(1,1e-2, size=(self.out_dim,self.hid_dim+1))
        )
    def forward(self, X):
        Hin = self.W @ X
        _, num_samples = Hin.shape
        #Hin = np.concatenate((Hin, np.ones((1,num_samples))), axis=0)
        #Hout = sig(Hin)
        #dsig1 = dsig(Hin)
        Hout = sig(Hin)
        Hin = np.concatenate((Hout, np.ones((1,num_samples))), axis=0)#will be removed later in train
        dsig1 = dsig(Hin)
        Hout = np.concatenate((Hout, np.ones((1,num_samples))), axis=0)
        # add ow for bias
        Oin = self.V @ Hout
        Oout = sig22(Oin)
        dsig2 = dsig22(Oin)
        #return Hin, Hout, dsig1, Oin, Oout, dsig2
        return Hout, dsig1, Oout, dsig2

    def dec_boundary(self, x_start, x_stop, y_start, y_stop, interval):
        nx = int((x_stop-x_start)/interval + 1)
        ny = int((y_stop-y_start)/interval + 1)
        x_lim = np.linspace(x_start,x_stop, nx)
        y_lim = np.linspace(y_start,y_stop, ny)
        #xv, yv = np.meshgrid(x, y)
        data = []
        for x in x_lim:
            for y in y_lim:
               data.append([x,y])
        data = np.array(data).T
        #xyv = np.concatenate((xv,yv),axis = 0)
        data = np.concatenate((data,np.ones((1,data.shape[1]))),axis=0)

        #print(xv)
        #print(yv)
        #print(xyv)
        # TODO>
        # TODO: get indices from O that fulfills condition
        # then take points from dat, and returb
        #print("data shape",data.shape)
        _,_,O,_ = self.forward(data)
        print("O",O)
        dec_threshold = 1e-7
        point_idx = np.where(np.abs(O) < dec_threshold)[1]
        print("indices",point_idx.shape)
        #print("-----------")
        return data[:,point_idx.T]
        #print(O.shape)
        #print(dec_bound)
        #print(dec_bound[1])

    def train(self, config):
        X = config['data']
        T = config['labels']
        X_val = config['data_val']
        T_val = config['labels_val']
        update_type = config['update_type']
        thres = config['threshold']
        eta = config['learn_rate']
        a = config['alpha']

        _, N = X.shape
        msg_epoch = []
        class_errs = []
        _, N_val = X_val.shape
        msg_epoch_val = []
        class_errs_val = []


        # TODO: is this assumption correct?
        theta = 0.
        psi = 0.
        msg = 10000 

        # initial forward
        H, dsig_hid, O, dsig_out = self.forward(X)
        for epoch in range(config['epochs']):
            # TODO: fix sequential
            
            # backward (forward is already done)
            Odelta = np.multiply((O - T), dsig_out) 
            Hdelta = np.multiply((self.V.T @ Odelta), dsig_hid)
            Hdelta = np.delete(Hdelta,-1,0)

            theta = a*theta - (1-a)*Hdelta @ X.T
            psi = a*psi - (1-a)*Odelta @ H.T

            self.W += eta*theta
            self.V += eta*psi
            '''
            print("------------")
            print("W weight: ",self.W)
            print("V weight: ",self.V)
            '''

            '''
            if np.sum(np.abs(old_W - self.W)) < thres:
                print('converged at epoch: {}'.format(epoch))
                return epoch, msg_epoch
            '''
            # log mean square error in validation
            _, _, O_val,_ = self.forward(X_val)
            e_val = (T_val - O_val)
            msg_val = e_val @ e_val.T * 1/N_val
            msg_val = msg_val[0,0]
            msg_epoch_val.append(msg_val)
            #print("MSE val epoch ",epoch, ": ", msg_val)
            
            # ratio of missclassification in validation
            class_err_val = 1 - np.sum(np.vectorize(classify)(O_val,T_val)) / N_val
            class_errs_val.append(class_err_val)
            
            # log mean square error in training set
            e = (T - O)
            old_msg = msg
            msg = e @ e.T * 1/N
            msg = msg[0,0]
            msg_epoch.append(msg)
            
            # ratio of missclassification training
            H, dsig_hid, O, dsig_out = self.forward(X)
            class_err = 1 - np.sum(np.vectorize(classify)(O,T)) / N
            class_errs.append(class_err)

            # TODO: this thresholding doesnt work for s-curve
            #print(np.abs(msg- old_msg)) 
            #if np.abs(msg - old_msg) < config['threshold']:
            #    print('converged at epoch: {}'.format(epoch))
            #    return epoch, msg_epoch
        
        return config['epochs'], msg_epoch, class_errs, msg_epoch_val, class_errs_val


# end net2 class 

# returns 1 if correctly classified, otherwise 0
def classify(o, t):
    return int((o > 0 and t == 1) or (o <= 0 and t == -1))

def plot_errs(config, errs, errs_val, network, err_label='mse'): 
    plt.plot(errs, 'g')
    plt.plot(errs_val,'r')
    plt.title('{} error for MLP with {} update function, with learning rate {}:'.format(
        err_label,
        config['update_type'], 
        config['learn_rate'],
    ))
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.show()
   # plt.savefig(
   #     'errs/mse_funcapprox_lr_{}_hid_{}.png'.format(
   #         config['learn_rate'],network.hid_dim
   # ))
 
'''
def plot_err(config, errs, network, err_label='mse'): 
    plt.plot(errs)
    plt.title('{} error for MLP with {} update function, with learning rate {}:'.format(
        err_label,
        config['update_type'], 
        config['learn_rate'],
    ))
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.savefig(
        'errs/{}_mlp_{}_lr_{}_epochs_{}_hid_{}.png'.format(
            err_label,config['update_type'], config['learn_rate'], 
            config['epochs'], network.hid_dim
         ))

    plt.show()
'''      
# batch X,T inputs and targets each column
# W = {w_ji} weight from x_i to t_j
# adds the negative gradient to minimize error
def delta_update_batch(W,X,T,eta):
    return W - eta*(W @ X - T) @ X.T

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

def sig22(x):
    return x
def dsig22(x):
    return 1

def sig(x):
    #return x
    return 2. / (1. + np.exp(-2*x)) - 1.

def dsig(x):
    #return 1
    #e = np.exp(-0.3*x)
    return 1 - sig(x)*sig(x)
    #return 20. * (0.3*e)/((1+e)*(1+e))
    #return (1. + sig(x))*(1. - sig(x)) / 2.

def data_gen(N, m_a, m_b, sig_a, sig_b, x_dim=2, out_dim=1):
    N = int(N)
    class_a = np.random.multivariate_normal(m_a, sig_a, N)
    class_b = np.random.multivariate_normal(m_b, sig_b, N)
    T_a = np.ones((out_dim,N))
    T_b = -1*np.ones((out_dim,N))
    T = np.concatenate((T_a,T_b), axis=1)
    data = np.concatenate((class_a, class_b), axis=0).T
    return data, T

def data_gen_subsample(N, m_a, m_b, sig_a, sig_b, x_dim=2, out_dim=1,exp=4):
    N = int(N)
    class_a = np.random.multivariate_normal(m_a, sig_a, N).T
    T_a = np.ones((out_dim,N))
    a = np.concatenate((class_a,T_a), axis=0)
    _,N_a = a.shape             

    class_b = np.random.multivariate_normal(m_b, sig_b, N).T
    T_b = -1*np.ones((out_dim,N))
    b = np.concatenate((class_b, T_b), axis=0)
    _,N_b = a.shape             

    if exp == 1:
        N_a25 = int(N_a * 0.25)
        N_b25 = int(N_b * 0.25)
        a_val = a[:,0:N_a25]
        a_train = a[:,N_a25:]
        b_val = b[:,0:N_b25]
        b_train = b[:,N_b25:]

        train = np.concatenate((a_train,b_train), axis=1)
        val = np.concatenate((a_val,b_val), axis=1)
    if exp == 2:
        N_a50 = int(N_a * 0.5)
        a_val = a[:,0:N_a50]
        a_train = a[:,N_a50:]
        #b_val = b[:,0:N_b25]
        #b_train = b[:,N_b25:]

        train = np.concatenate((a_train,b), axis=1)
        val = a_val
    if exp == 3:
        N_b50 = int(N_b * 0.5)
        b_val = b[:,0:N_b50]
        b_train = b[:,N_b50:]
        #b_val = b[:,0:N_b25]
        #b_train = b[:,N_b25:]

        train = np.concatenate((b_train,a), axis=1)
        val = b_val
    if exp == 4:
        a_sub1 = []
        a_sub2 = []
        for col in a.T:
            if col[0] < 0:
                a_sub1.append(col)
            else:
                a_sub2.append(col)

        N_asub1 = len(a_sub1)
        N_asub2  = len(a_sub2)
        N_sub1_20 = int(float(N_asub1)*0.2)
        N_sub2_80 = int(float(N_asub2)*0.8)

        a_sub1 = np.array(a_sub1).T
        a_sub2 = np.array(a_sub2).T

        # 20 % of <0 and 80% of 0<
        class_a_val = np.concatenate(
            (a_sub1[:,0:N_sub1_20], a_sub2[:,0:N_sub2_80]),
            axis=1)
        # the rest for training
        class_a_train = np.concatenate(
            (a_sub1[:,N_sub1_20:], a_sub2[:,N_sub2_80:]),
            axis=1)
        # put rest and class b together
        train = np.concatenate((class_a_train, b),axis=1)
        val = class_a_val
    data_train = train[:2,:]
    T_train = train[2,:]
    data_val = val[:2,:]
    T_val = val[2,:]
    
    return data_train, T_train, data_val, T_val


def plot_data_func(config, network, line_start, line_end, convergence_epoch, xx,yy,z):
    data = config['data_val']
    labels = config['labels_val']
    Hout, _, out, _ = network.forward(data) 
    print("out",out)
    print("val",labels)

    data = np.delete(data, (-1), axis=0)
    x, y = data

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(x,y,out, c='r')
    ax.scatter(x,y,labels, c='b')
    #surf = ax.plot_surface(xx,yy,z)
    plt.show()
    #plt.savefig('plots/func_lr_{}_hid_{}.png'.format(
    #    config['learn_rate'],network.hid_dim
    #))

def main():
    # set seed
    np.random.seed(1234)

    # config layers
    x_dim = 2
    out_dim = 1
    h_dim = 25
    N = 400

    val_portion = 0.2

    # data generation real func
    x = np.arange(-5., 5., 0.5)
    y = np.arange(-5., 5., 0.5)
    xx, yy = np.meshgrid(x, y)
    z = np.exp(-xx**2/10) * np.exp(-yy**2/10)-0.5
    #print("ZZZZZ",z)
    #z = np.exp(-xx.dot(xx.T)/10) * np.exp(-yy.dot(yy.T)/10)-0.5
    '''
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xx,yy,z)
    '''
    labels = np.reshape(z,(1,400))
    patterns1 = np.reshape(xx,(1,400))
    patterns2 = np.reshape(yy,(1,400))
    patterns = np.concatenate((patterns1, patterns2), axis = 0)
    patterns = np.concatenate((patterns, np.ones((1,400))), axis=0)
    pnl = np.concatenate((patterns,labels),axis=0).T
    np.random.shuffle(pnl)
    pnl = pnl.T
    patterns_val = pnl[0:3,0:int(400*val_portion)]
    patterns_train = pnl[:3,int(400*val_portion):400]
    labels_val = pnl[3:,:int(400*val_portion)]
    labels_train = pnl[3:,int(400*val_portion):]
    #print(labels_val.shape)
    #print(labels_train.shape)
    
    config = {
        'data': patterns_train,
        'labels': labels_train,
        'data_val': patterns_val,
        'labels_val': labels_val,
        'epochs': 100,
        'learn_rate': 0.001,
        'update_type': 'batch',
        'threshold': 1e-9,
        'alpha': 0.9,
        'save_fig': False
    }
    # Network objects
    net2 = Net2(x_dim, h_dim,out_dim)
    print("Net initialised ... ")

    # training config
    alpha = 0.9
    convergence_epoch, msg_epoch, class_errs, msg_epoch_val, class_errs_val = net2.train(config)
    #convergence_epoch, msg_epoch = net1.train(config)
    print("Training finished ...")

    #create plots
    plot_data_func(config, net2, -6, 10, convergence_epoch,xx,yy,z)
    plot_errs(config, msg_epoch,msg_epoch_val, net2, 'mse')

if __name__ == "__main__":
    main()

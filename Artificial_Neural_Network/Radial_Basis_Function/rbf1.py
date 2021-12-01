import numpy as np
import matplotlib.pyplot as plt

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
            np.random.normal(1,1e-2, size=(self.hid_dim,self.in_dim+1)),
            np.random.normal(1,1e-2, size=(self.out_dim,self.hid_dim+1))
        )
    def forward(self, X):
        Hin = self.W @ X
        _, num_samples = Hin.shape

        Hout = sig(Hin)
        Hin = np.concatenate((Hout, np.ones((1,num_samples))), axis=0)
        dsig1 = dsig(Hin)
        Hout = np.concatenate((Hout, np.ones((1,num_samples))), axis=0)

        Oin = self.V @ Hout
        Oout = sig22(Oin)
        dsig2 = dsig22(Oin)
        return Hout, dsig1, Oout, dsig2

    # train and collect train,test mse
    def train(self, pat, lab, pat_test, lab_test, eta, epochs):
        N = pat.shape[0]
        X = pat.copy().reshape((N,1)).T
        X = np.concatenate((X, np.ones((1,N))), axis=0)
        X_test = pat_test.copy().reshape((N,1)).T
        X_test = np.concatenate((X_test, np.ones((1,N))), axis=0)

        T = lab.copy().T
        T_test = lab_test.copy().T
        
        a = 0.9

        mse_epoch = []
        mse_epoch_test = []

        theta = 0.
        psi = 0.

        # initial forward
        H, dsig_hid, O, dsig_out = self.forward(X)
        for epoch in range(epochs):
            # backward (forward is already done)
            Odelta = np.multiply((O - T), dsig_out)
            Hdelta = np.multiply((self.V.T @ Odelta), dsig_hid)
            Hdelta = np.delete(Hdelta,-1,0)

            theta = a*theta - (1-a)*Hdelta @ X.T
            psi = a*psi - (1-a)*Odelta @ H.T

            self.W += eta*theta
            self.V += eta*psi
            
            # log mean square error in training set
            e = (T - O)
            mse = e @ e.T * 1/N
            mse = mse[0,0]
            mse_epoch.append(mse)

            # log mean square error in test
            _, _, O_test,_ = self.forward(X_test)
            e_test = (T_test - O_test)
            mse_test = e_test @ e_test.T
            mse_test = mse_test[0,0]
            mse_epoch_test.append(mse_test)

            return mse_epoch, mse_epoch_test 

def sig(x):
    #return x
    return 2. / (1. + np.exp(-2*x)) - 1.

def dsig(x):
    #return 1
    #e = np.exp(-0.3*x)
    return 1 - sig(x)*sig(x)
    #return 20. * (0.3*e)/((1+e)*(1+e))
    #return (1. + sig(x))*(1. - sig(x)) / 2.

def sig22(x):
    return x

def dsig22(x):
    return 1

class RBF:
    def __init__(self):
        #self.h_dim = 9
        self.means = np.arange(0,2*np.pi,np.pi/4)
        #self.means = 2*np.pi*np.random.rand(16)
        self.h_dim = self.means.shape[0]
        print("hidden: ", self.h_dim)
        max_dist = 0.5*np.pi
        self.std = max_dist / np.sqrt(2*self.h_dim)
        print("std: ",self.std)
        #self.means = np.array([0,np.pi/4, 2*np.pi/4,3*np.pi/4,4*np.pi/4,
        #     5*np.pi/4 ,6*np.pi/4, 7*np.pi/4,8*np.pi/4])
        #self.means = np.array([0,np.pi/4 ,3*np.pi/4, 5*np.pi/4, 7*np.pi/4,2*np.pi])
        self.weights = np.random.normal(0,0.1,self.h_dim).T # row vector

    def rbfunc(self, x,idx):
        mean = self.means[idx]
        std = self.std
        #denom = np.sum(-(x-self.means)*(x-self.means)/(2*std*std))
        return np.exp(-(x-mean)*(x-mean)/(2*std*std)) #/ denom

    def forward(self,X):
        N = X.shape[0]
        R = np.empty([N,self.h_dim])
        for x_idx in range(N):
            for r_idx in range(self.h_dim):
                x = X[x_idx]
                print("x shape: ",x.shape)
                print("rbf shape: ",self.rbfunc(x,r_idx).shape)
                R[x_idx,r_idx] = self.rbfunc(x,r_idx)
        return R,R @ self.weights

    def forward_seq(self,x):
        R = np.empty([self.h_dim,1]) # row vect
        for r_idx in range(self.h_dim):
            R[r_idx] = self.rbfunc(x,r_idx)
        return R,self.weights.T.dot(R)[0]

    # uses lst square
    def train(self, patterns, labels):
        A,_ = self.forward(patterns)
        self.weights = np.linalg.lstsq(A,labels,rcond=None)[0]

    # uses seq. delta
    def train_seq(self,patterns,labels,learn_rate,epochs):
        N = patterns.shape[0]
        # reshape (N,) to (N,1)
        X_shuff = patterns.reshape((N,1))
        T_shuff = labels.reshape((N,1))

        for epoch in range(epochs):
            # shuffle data
            points = np.concatenate((X_shuff,T_shuff), axis=1)
            np.random.shuffle(points)
            X_shuff = points[:,0].reshape((N,1)) 
            T_shuff = points[:,1].reshape((N,1)) 
            for i in range(N):
                x = X_shuff[i]
                t = T_shuff[i]
                R_col,out = self.forward_seq(x)
                e = np.linalg.norm(t-out)
                #print("weight shape: ",self.weights.shape)
                #print("R_col shape: ",R_col.shape)
                delta = learn_rate*e*R_col[:,0]
                #print("delt max = ",np.max(delta))
                #print("delt min = ",np.min(delta))
                #print("----")
                self.weights += delta

    def test(self, pat, lab):
        N = pat.shape[0]
        labels = lab.copy().T
        patterns = pat.copy().reshape((N,1)).T
        patterns = np.concatenate((patterns, np.ones((1,N))), axis=0)
        _,out = self.forward(patterns)
        e = np.linalg.norm(out - labels)
        mse = e*e
        #avg_residual = np.average(e)
        #return avg_residual, out 
        return mse, out 

    def test_transform(self, patterns, labels):
        _,out = self.forward(patterns)
        e = np.linalg.norm(out - labels)
        mse = e*e
        #avg_residual = np.average(e)
        out_transform = np.piecewise(out, [out >= 0, out < 0], [1,-1])
        e_transform = np.linalg.norm(out_transform - labels)
        mse_trans = e_transform*e_transform
        #avg_residual_trans = np.average(e_transform)
        #return avg_residual, out, out_transform, avg_residual_trans 
        return mse, out, out_transform, mse_trans 

def sin(x):
    return np.sin(x)
def square(x):
    return np.piecewise(x, [sin(x) >= 0, sin(x) < 0], [1,-1])

def main():
    X_train = np.arange(0,2*np.pi, 0.1)
    N_train = X_train.shape[0]
    noise_train = np.random.normal(0,np.sqrt(0.1), N_train)
    X_train += noise_train

    X_test = np.arange(0.05,2*np.pi, 0.1)
    N_test = X_test.shape[0]
    noise_test = np.random.normal(0,np.sqrt(0.1), N_test)
    X_test += noise_test

    labels1_train = sin(2*X_train)
    labels2_train = square(2*X_train)
    labels1_test = sin(2*X_test)
    labels2_test = square(2*X_test)

    rbf11 = RBF()
    rbf12 = RBF()
    h_dim = rbf11.h_dim
    mlp1 = MLP(1, h_dim ,1)
    rbf21= RBF()
    rbf22= RBF()
    mlp2 = MLP(1, h_dim ,1)

    learn_rate = 0.0005
    learn_rate_mlp = 0.001
    epochs = 20
    epochs_mlp = 200

    # sin 2x
    rbf11.train(X_train,labels1_train)
    rbf12.train_seq(X_train,labels1_train, learn_rate, epochs)
    mse_train1, mse_test1 = mlp1.train(
        X_train, labels1_train, X_test,
        labels1_test, learn_rate_mlp, epochs_mlp)
    err11, out_test11 = rbf11.test(X_test,labels1_test)
    err12, out_test12 = rbf12.test(X_test,labels1_test)
    err13, out_test13 = mlp1.test(X_test, labels1_test)

    # errors sin2x
    #print("Weights rbf11: ", rbf11.weights)
    print("Err11: ",err11)
    #print("Weights rbf12: ", rbf12.weights)
    print("Err12: ",err12)
    
    #print("weights rbf12 seq: ", rbf12.weights)

    #plot figs
    fig, (ax1,ax2,ax3,ax4) = plt.subplots(4)
    fig.suptitle("sin2x batch/seq, square2x batch/seq")
    # plot batch for sin2x
    ax1.plot(X_test, labels1_test, 'g')
    ax1.plot(X_test, out_test11, 'r')
    ax2.plot(X_test, labels1_test, 'g')
    ax2.plot(X_test, out_test12, 'r')

    # square fnc
    rbf21.train(X_train,labels2_train)
    rbf22.train_seq(X_train,labels2_train, learn_rate, epochs)

    err21,out_test21,out21_trans,trans_err21 = \
        rbf21.test_transform(X_test,labels2_test)
    err22,out_test22,out22_trans,trans_err22 = \
        rbf22.test_transform(X_test,labels2_test)
    print("Err21: ",err21)
    print("Err22: ",err22)
    print("Transformed err 21: ", trans_err21)
    print("Transformed err 22: ", trans_err22)

    ax3.plot(X_test, labels2_test, 'g')
    ax3.plot(X_test, out21_trans, 'r')
    ax4.plot(X_test, labels2_test, 'g')
    ax4.plot(X_test, out22_trans, 'r')
    plt.show()
    
main()

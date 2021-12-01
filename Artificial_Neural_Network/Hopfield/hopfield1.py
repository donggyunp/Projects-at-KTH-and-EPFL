import numpy as np
import matplotlib.pyplot as plt
import itertools

class Hopfield:
    def __init__(self, dim):
        self.dim = dim
        a = np.tril(np.random.rand(dim,dim)) # 0-1
        # symmetric weights
        self.W = np.zeros((dim,dim))

    def random_weights(self):
        self.W = np.random.normal(0,0.1,self.dim*self.dim)
        self.W = self.W.reshape(self.dim,self.dim)

    def symmetric_weights(self):
        self.W = np.tril(self.W) + np.triu(self.W.T, 1) 

    def random_sym_weights(self):
        self.W = np.random.normal(0,0.1,self.dim*self.dim)
        self.W = self.W.reshape(self.dim,self.dim)

    def train(self, patterns):
        N,_ = patterns.shape
        #print(patterns.shape)
        for i in range(N):
            self.W += np.outer(patterns[i,:].T,patterns[i,:]) 
            #print("Weights: ",self.W.shape,"\n",self.W)
        np.fill_diagonal(self.W,0)

    def train_nonsym(self, patterns):
        N,_ = patterns.shape
        print(patterns.shape)
        for i in range(N):
            self.W += np.outer(patterns[i,:].T,patterns[i,:]) 
            #print("Weights: ",self.W.shape,"\n",self.W)
    
    def train_bias(self, patterns):
        N,_ = patterns.shape
        for i in range(N):
            self.W += np.outer(patterns[i,:].T,patterns[i,:])
            #print("Weights: ",self.W.shape,"\n",self.W)
        np.fill_diagonal(self.W,0)

    def train_sparse(self, patterns, self_connect=False):
        P,_ = patterns.shape
        #imagine rho
        rho = np.sum(np.sum(patterns))/(self.dim * P)
        print("rho!!!!!!!!!!!!!!!:", rho)
        for i in range(P):
            self.W += np.outer( (patterns[i,:]-rho).T, (patterns[i,:]-rho) )
        if not self_connect:
            np.fill_diagonal(self.W,0)    


    def recall(self, patterns):
        X = patterns.copy()
        N,x_dim = patterns.shape
        num_attr = np.zeros(N)
        lis = []
        #print("NNNNN",N)
        for i in range(N):
            print("point ",X[i])
            for e in range(100):
                old = X[i].copy()
                X[i] = np.sign(self.W @ X[i])
                if np.all(X[i]==old):
                    print("iterations: ",e)
                    num_attr[i] += 1
                    lis.append(X[i])
                    break
            print("result: ",X[i])
        print("done")
        print(X)
        #print(num_attr)
        return X,np.array(lis)

    def energy_recall(self, patterns, epochs=50, bias=False):
        X = patterns.copy()
        N,x_dim = patterns.shape
        num_attr = np.zeros(N)
        lis = []
        energies = []
        for i in range(N):
            energy = [self.energy(X[i])]
            for e in range(epochs):
                old = X[i].copy()
                if bias:
                    X[i] = np.sign(self.W @ X[i] + 0.5)
                else:
                    X[i] = np.sign(self.W @ X[i])
                e = self.energy(X[i])
                energy.append(e)
                if self.energy(old) == self.energy(X[i]):
                    #print("energy stop iter ",e,": ",e)
                    break
            energies.append(energy)
        return X,energies

    def energy_recall_sparse(self, patterns, epochs=50, bias=0):
        X = patterns.copy()
        N,x_dim = patterns.shape
        num_attr = np.zeros(N)
        lis = []
        energies = []
        for i in range(N):
            energy = [self.energy(X[i])]
            for e in range(epochs):
                old = X[i].copy()
                X[i] = 0.5 + 0.5*np.sign(self.W @ X[i] + bias)
                e = self.energy(X[i])
                energy.append(e)
                if self.energy(old) == self.energy(X[i]):
                    #print("energy stop iter ",e,": ",e)
                    break
            energies.append(energy)
        return X,energies


    def recall_rand(self, patterns, epochs, printing):
        X = patterns.copy()
        N,x_dim = patterns.shape
        num_attr = np.zeros(N)
        lis = []
        print("NNNNN",N)
        for i in range(N):
            print("point ",X[i])
            for e in range(epochs):
                old = X[i].copy()
                random = np.random.choice(1024,1)
                X[i,random] = np.sign(np.dot(self.W[random], X[i]))
                if e%printing == 0 or e == epochs-1:
                    fig, axs = plt.subplots()
                    plot = X[i,:].reshape((32,32))
                    fig.suptitle('The iteration:' + str(e) +'\n which unit?'+ str(i))
                    plt.imshow(plot)
                plt.show()
    
                '''
                if np.all(X[i]==old):
                    print("iterations: ",e)
                    num_attr[i] += 1
                    lis.append(X[i])
                    break
                '''
            #print("result: ",X[i])
        print("done")
        #print(X)
        #print(num_attr)
        return X#,np.array(lis)

    def energy(self, x):
        E = 0
        for i in range(self.dim):
            for j in range(self.dim):
                E -= self.W[i,j]*x[i]*x[j]
        return E
'''
def plotting(original,num_row):
    fig, axs = plt.subplots(num_row)
    plot = []
    for i in range(num_row):
        plot = original[i,:].reshape((32,32))
        axs[0].imshow(plot)
        plot = recall_rand(original)[i,:].reshape((32,32))
        axs[1].imshow(plot)
    plt.show()
'''
def generate_all():
    permutate = list(itertools.product([-1,1], repeat=8))
    permutate = np.array(permutate)
    return permutate
            
def get_data():
    pic = []
    with open('pict.dat', 'r') as f:
        d = f.readlines()[0].split(',')
        for m in range(9):
            row = []
            for v in range(1024):
                row.append(float(d[m*1024 + v]))
            pic.append(row) 
        
    return np.array(pic)

def experiment1():
    X_real = np.array([[-1,-1,1,-1,1,-1,-1,1],
                      [-1,-1,-1,-1,-1,1,-1,-1],
                      [-1,1,1,-1,-1,1,-1,1]])
    X_dist = np.array([[1,-1,1,-1,1,-1,-1,1],
                      [1,1,-1,-1,-1,1,-1,-1],
                      [1,1,1,-1,1,1,-1,1]])
    X_more_dist =  np.array([[1,1,1,1,1,1,1,1],
                             [1,1,1,1,1,1,-1,-1],
                             [-1,-1,-1,-1,-1,-1,-1,-1]])
    
    
    num_units = 8
    hop1 = Hopfield(num_units)
    hop1.train(X_real)
    #res = hop1.recall(X_more_dist)
    cases = generate_all()
    allcases, attractors = hop1.recall(cases)
    unique = np.unique(attractors, axis = 0)
    print(unique.shape)
   
def experiment2():
    num_units = 1024
    hop1 = Hopfield(num_units)
    X_real = get_data()
    hop1.train(X_real[0:3,:])

    recalled, _ = hop1.recall(X_real[0:3,:])

    #change row number to see different plots
    for i in range(3):
    #for i in range(X_real.shape[0]):
        fig, axs = plt.subplots(2)
        plot = X_real[i,:].reshape((32,32))
        fig.suptitle('The original image and recalled image')
        axs[0].imshow(plot)
        plot = recalled[i,:].reshape((32,32))
        axs[1].imshow(plot)
    plt.show()

    #degrade p1
    p10 = X_real[0,:].copy()
    p10 = p10.reshape((1,1024))
    select = np.random.choice(1024,200)
    for i in select:
        p10[0,i] = 1
    p10_recall,_ = hop1.recall(p10)
    
    fig, axs = plt.subplots(3)
    fig.suptitle('300 pixels degraded p1 and recall')
    plot = X_real[0,:].reshape((32,32))
    axs[0].imshow(plot)
    plot = p10[0,:].reshape((32,32))  
    axs[1].imshow(plot)
    plot = p10_recall[0,:].reshape((32,32))
    axs[2].imshow(plot)
    plt.show()

    
    _ = hop1.recall_rand(p10,10000,1000)

def experiment3():
    num_units = 1024
    hop1 = Hopfield(num_units)
    X_real = get_data()
    hop1.train(X_real[0:3,:])

    # get energies at attractors
    e_p1 = hop1.energy(X_real[0,:])
    e_p2 = hop1.energy(X_real[1,:])
    e_p3 = hop1.energy(X_real[2,:])
    print("Energies at attractors: {},{},{}".format(e_p1,e_p2,e_p3))
    
    #degrade p1
    p10 = X_real[1,:].copy()
    p10 = p10.reshape((1,1024))
    select = np.random.choice(1024,200)
    for i in select:
        p10[0,i] = 1
    # p10_recall,energies = hop1.energy_recall(p10)
    e_p10 = hop1.energy(p10[0])
    print("Energy at p10: ", e_p10)

    '''
    recall_p10,energies = hop1.energy_recall(p10,epochs=10)

    # plot energies
    plt.plot(energies[0]) 
    # plot images
    
    fig, axs = plt.subplots(2)
    plot_p10 = p10[0].reshape((32,32))
    plot_p10_recall = recall_p10[0].reshape((32,32))
    fig.suptitle('The original and recalled image')
    axs[0].imshow(plot_p10)
    axs[1].imshow(plot_p10_recall)
    '''

    # set weights to normally distributed
    hop1.random_weights()
    #hop1.train(X_real[0:3,:])
    hop1.train_nonsym(X_real[0:3,:])
    #hop1.symmetric_weights() 

    recall_p10,energies = hop1.energy_recall(p10,epochs=7)
    print("Noisy energy recalled: ",energies[0])
    fig, axs = plt.subplots(2)
    plot_p10 = p10[0].reshape((32,32))
    plot_p10_recall = recall_p10[0].reshape((32,32))
    fig.suptitle('The original and recalled image w random weighrs')
    axs[0].imshow(plot_p10)
    axs[1].imshow(plot_p10_recall)

    fig,axs = plt.subplots()
    axs.plot(energies[0])
     
    plt.show()

def rand_flip(x):
    p = np.random.rand() 
    if p <= 0.8:
        return x * -1
    return x

def experiment4():
    num_units = 1024
    hop1 = Hopfield(num_units)
    X_real = get_data()
    hop1.train(X_real[0:3,:])

    #degrade p1,p2,p3
    p10 = X_real[0,:].copy()
    p11 = X_real[1,:].copy()
    p12 = X_real[2,:].copy()
    num_rand = int(1024*0.7)
    noise = np.ones(1024)
    select1 = np.random.choice(1024,int(num_rand*0.5))
    select2 = np.random.choice(1024,int(num_rand*0.5))
    noise[select1] *= -1
    noise[select2] *= -1
    p10 *= noise
    p11 *= noise
    p12 *= noise
   
    data = np.array([p10,p11,p12])

    # recall and plot
    recalled,_ = hop1.energy_recall(data, epochs=10)
    for i in range(3):
        fig, axs = plt.subplots(2)
        plot = data[i].reshape((32,32))
        fig.suptitle('The original image and recalled image')
        axs[0].imshow(plot)
        plot = recalled[i,:].reshape((32,32))
        axs[1].imshow(plot)
    plt.show()

def experiment53end():
    num_units = 1024
    hop1 = Hopfield(num_units)
    X_real = get_data()
    hop1.train(X_real[0:3,:])
    print(X_real[5,:].shape)
    hop1.train(X_real[5,:].reshape(1,1024))
    p_1235 =np.concatenate((X_real[0:3,:],X_real[5,:].reshape(1,1024)),axis=0)
    recalled, _ = hop1.recall(p_1235)
    
    for i in range(4):
    #for i in range(X_real.shape[0]):
        fig, axs = plt.subplots(2)
        plot = p_1235[i,:].reshape((32,32))
        fig.suptitle('The original image and recalled image from p1 to p4')
        axs[0].imshow(plot)
        plot = recalled[i,:].reshape((32,32))
        axs[1].imshow(plot)
    plt.show()

    noise = np.ones(1024)
    select1 = np.random.choice(1024,500)
    select2 = np.random.choice(1024,500)
    noise[select1] *= -1
    noise[select2] *= -1
   
    hop2 = Hopfield(num_units)
    concat = np.concatenate((X_real[0:3,:],noise.reshape(1,1024)),axis=0)
    hop2.train(concat)
    recalled2, _ = hop2.recall(concat)
    
    for i in range(4):
    #for i in range(X_real.shape[0]):
        fig, axs = plt.subplots(2)
        plot = concat[i,:].reshape((32,32))
        fig.suptitle('The original image and recalled image from p1 to p4')
        axs[0].imshow(plot)
        plot = recalled2[i,:].reshape((32,32))
        axs[1].imshow(plot)
    plt.show()

def experiment54begin():
    
    num_units = 100
    data = np.zeros((300,100))

    for i in range(300):
        noise = np.ones(100)
        select1 = np.random.choice(100,50)
        select2 = np.random.choice(100,50)
        noise[select1] *= -1
        noise[select2] *= -1
        data[i] = noise

    hop1=Hopfield(num_units)
    stables = []
    for i in range(100):#range(300):
        d_train = data[i,:].reshape(1,100)
        #hop1.train_nonsym(d_train)
        hop1.train(d_train)
        d_test = data[0:i+1,:]#.reshape(i+1, 100)
        recalled,_ = hop1.energy_recall(d_test, epochs=1)
        #recalled,_ = hop1.energy_recall(d_test, epochs=1, bias=True)

        num_stable = i+1
        for j in range(i+1):
            if np.any(recalled[j] != data[j]):
                    num_stable -= 1
        print("Trained on ",i," points, num stable: ",num_stable)
        stables.append(num_stable)
    #stables = [1,2,3,4,5,6,7,7,8,8,8,9,9,9,6,6,3,5,6,3,5,4,3,2,1,1,1,0,0,1,0,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    stables_plot = np.array(stables)
    plt.plot(stables_plot)
    plt.show()

def experiment6():
    grid = 20
    num_unit = grid*grid

    it = 1
    data = np.zeros((it,num_unit))

    for i in range(it):
        noise = np.zeros(num_unit)
        select1 = np.random.choice(num_unit,50)
        noise[select1] += 1
        data[i] = noise
    fig, axs = plt.subplots(2)
    plot = data[0,:].reshape((grid,grid))
    axs[0].imshow(plot)
    plot = data[1,:].reshape((grid,grid))
    axs[1].imshow(plot)
    plt.show()
    hop1=Hopfield(num_unit)
    stables = []
    bias = [0, 0.25, 0.5, 0.75, 0.9]
    for k in bias:
        for i in range(it):#range(300):
            d_train = data[i,:].reshape(1,num_unit)
            #hop1.train_nonsym(d_train)
            hop1.train_sparse(d_train)
            d_test = data[0:i+1,:]#.reshape(i+1, 100)
            recalled,_ = hop1.energy_recall_sparse(d_test, epochs=1,bias=k)
            #recalled,_ = hop1.energy_recall(d_test, epochs=1, bias=True)

            num_stable = i+1
            for j in range(i+1):
                if np.any(recalled[j] != data[j]):
                    num_stable -= 1
            print("Trained on ",i," points, num stable: ",num_stable)
            stables.append(num_stable)
        stables_plot = np.array(stables)
        plt.plot(stables_plot)
        plt.title('sparse patterns with bias = {}'.format(k))
        plt.show()

    

def main():
    #experiment1()    
    #experiment2()
    #experiment3()
    #experiment4()
    experiment53end()
    #experiment54begin()
    #experiment6()
    
main()

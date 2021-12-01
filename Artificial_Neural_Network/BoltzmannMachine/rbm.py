from util import *

class RestrictedBoltzmannMachine():
    '''
    For more details : A Practical Guide to Training Restricted Boltzmann Machines https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
    '''
    def __init__(self, ndim_visible, ndim_hidden, is_bottom=False, image_size=[28,28], is_top=False, n_labels=10, batch_size=10):

        """
        Args:
          ndim_visible: Number of units in visible layer.
          ndim_hidden: Number of units in hidden layer.
          is_bottom: True only if this rbm is at the bottom of the stack in a deep belief net. Used to interpret visible layer as image data with dimensions "image_size".
          image_size: Image dimension for visible layer.
          is_top: True only if this rbm is at the top of stack in deep beleif net. Used to interpret visible layer as concatenated with "n_label" unit of label data at the end. 
          n_label: Number of label categories.
          batch_size: Size of mini-batch.
        """
       
        self.ndim_visible = ndim_visible

        self.ndim_hidden = ndim_hidden

        self.is_bottom = is_bottom

        if is_bottom : self.image_size = image_size
        
        self.is_top = is_top

        if is_top : self.n_labels = 10

        self.batch_size = batch_size        
                
        self.delta_bias_v = 0

        self.delta_weight_vh = 0

        self.delta_bias_h = 0

        self.bias_v = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible))

        self.weight_vh = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible,self.ndim_hidden))

        self.bias_h = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_hidden))
        
        self.delta_weight_v_to_h = 0

        self.delta_weight_h_to_v = 0        
        
        self.weight_v_to_h = None
        
        self.weight_h_to_v = None

        self.learning_rate = 0.005
        
        self.momentum = 0.7

        self.print_period = 1
        
        self.rf = { # receptive-fields. Only applicable when visible layer is input data
            "period" : 5, # iteration period to visualize (5000)
            "grid" : [5,5], # size of the grid
            "ids" : np.random.randint(0,self.ndim_hidden,25) # pick some random hidden units
            }
        
        return

        
    def reconstruct(self,v0_batch):
        probs_h0_batch,h0_batch = self.get_h_given_v(v0_batch)
        probs_v1_batch,v1_batch = self.get_v_given_h(h0_batch)
        probs_h1_batch,h1_batch = self.get_h_given_v(v1_batch)
        probs_v2_batch,v2_batch = self.get_v_given_h(h1_batch)
        return v2_batch, h1_batch

    def networkEnergy(self, v, h):
        first = self.bias_v.reshape((1,self.bias_v.shape[0])) @ v.T
        second = self.bias_h.reshape((1,self.bias_h.shape[0])) @ h.T
        last = v @ self.weight_vh @ h.T
        return np.mean(- first - second - last)
        
    def cd1(self,visible_trainset, n_iterations=10000):
        
        """Contrastive Divergence with k=1 full alternating Gibbs sampling

        Args:
          visible_trainset: training data for this rbm, shape is (size of training set, size of visible layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        print ("learning CD1")
        
        n_samples = visible_trainset.shape[0]
        num_batch = n_samples // self.batch_size

        total_energies = []
        total_reconstruction_loss = []
        for it in range(n_iterations):
            epoch_energies = []
            for b in range(num_batch):
                train_batch = visible_trainset[b*self.batch_size : (b+1)*self.batch_size]
                # [TODO TASK 4.1] run k=1 alternating Gibbs sampling : v_0 -> h_0 ->  v_1 -> h_1.
                # you may need to use the inference functions 'get_h_given_v' and 'get_v_given_h'.
                # note that inference methods returns both probabilities and activations (samples from probablities) and you may have to decide when to use what.
                probs_h0_batch, h0_batch = self.get_h_given_v(train_batch)
                probs_v1_batch, v1_batch = self.get_v_given_h(h0_batch)
                probs_h1_batch, h1_batch = self.get_h_given_v(v1_batch)
                
                # [TODO TASK 4.1] update the parameters using function 'update_params'
                self.update_params(train_batch, h0_batch, probs_v1_batch, h1_batch)

                energy = self.networkEnergy(train_batch, h1_batch)
                epoch_energies.append(energy)

            # visualize once in a while when visible layer is input images
            #if it % self.rf["period"] == 0 and self.is_bottom:
            #    viz_rf(self.ndim_hidden, weights=self.weight_vh[:,self.rf["ids"]].reshape((self.image_size[0],self.image_size[1],-1)), it=it, grid=self.rf["grid"])

            # print progress
            if it % self.print_period == 0 :
                reconstruct_trainset, newest_hidden = self.reconstruct(visible_trainset)
                #energy = self.networkEnergy(visible_trainset, newest_hidden)
                #print("Energy: ",energy)
                print ("iteration=%7d recon_loss = %4.4f"%(it, np.linalg.norm(reconstruct_trainset - visible_trainset)))
            #print("Iteration=",it," Energy: ",np.mean(epoch_energies))
            # log energy and loss
            #total_energies.append(np.mean(epoch_energies))
            total_reconstruction_loss.append(np.linalg.norm(reconstruct_trainset - visible_trainset))
            print("Iteration=",it," Recon Loss: ",total_reconstruction_loss)

        '''
        fig, (ax1,ax2) = plt.subplots(2)
        fig.suptitle('Average energy and loss per Epoch')
        ax1.plot(total_energies)
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Energy')
        ax1.set_title('Energy plot:')
        ax2.plot(total_reconstruction_loss)
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.set_title('Loss plot:')
        plt.show()
        '''
        return
    

    def update_params(self,v_0,h_0,v_k,h_k):

        """Update the weight and bias parameters.

        You could also add weight decay and momentum for weight updates.

        Args:
           v_0: activities or probabilities of visible layer (data to the rbm)
           h_0: activities or probabilities of hidden layer
           v_k: activities or probabilities of visible layer
           h_k: activities or probabilities of hidden layer
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.1] get the gradients from the arguments (replace the 0s below) and update the weight and bias parameters
        self.delta_bias_v = self.learning_rate * (np.sum(v_0 - v_k, axis=0))
        self.delta_weight_vh = self.learning_rate * 1/self.batch_size * ((v_0.T @ h_0) - (v_k.T @ h_k))
        self.delta_bias_h = self.learning_rate * (np.sum(h_0 - h_k, axis=0))

        self.bias_v += self.delta_bias_v
        self.weight_vh += self.delta_weight_vh
        self.bias_h += self.delta_bias_h
        return

    def get_h_given_v(self,visible_minibatch):
        
        """Compute probabilities p(h|v) and activations h ~ p(h|v) 

        Uses undirected weight "weight_vh" and bias "bias_h"
        
        Args: 
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:        
           tuple ( p(h|v) , h) 
           both are shaped (size of mini-batch, size of hidden layer)
        """
        
        assert self.weight_vh is not None
        n_samples = visible_minibatch.shape[0]

        # [TODO TASK 4.1] compute probabilities and activations (samples from probabilities) of hidden layer (replace the zeros below) 
        #probs = np.zeros((n_samples,self.ndim_hidden))
        probs = sigmoid(self.bias_h + visible_minibatch @ self.weight_vh )
        assert (probs >= 0).all() and (probs <= 1).all()
        activations = sample_binary(probs) 
        return probs, activations 

    def get_v_given_h(self,hidden_minibatch):
        
        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses undirected weight "weight_vh" and bias "bias_v"
        
        Args: 
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:        
           tuple ( p(v|h) , v) 
           both are shaped (size of mini-batch, size of visible layer)
        """
        
        assert self.weight_vh is not None 
        n_samples = hidden_minibatch.shape[0]

        if self.is_top:

            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \ 
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """
            # [TODO TASK 4.2] compute probabilities and activations (samples from probabilities) of visible layer (replace the pass below). \
            # Note that this section can also be postponed until TASK 4.2, since in this task, stand-alone RBMs do not contain labels in visible layer.
            support = self.bias_v + hidden_minibatch @ self.weight_vh.T
            support_vis = support[:, self.n_labels:]
            support_labels = support[:, 0:self.n_labels]

            probs_vis = sigmoid(support_vis)
            probs_labels = softmax(support_labels)
            probs = np.concatenate((probs_labels, probs_vis),axis=1)
            assert (probs >= 0).all() and (probs <= 1).all()

            act_vis = sample_binary(probs_vis) 
            act_labels = sample_categorical(probs_labels) 
            activations = np.concatenate((act_labels, act_vis),axis=1)
            
        else:
            # [TODO TASK 4.1] compute probabilities and activations (samples from probabilities) of visible layer (replace the pass and zeros below)             
            probs = sigmoid(self.bias_v + hidden_minibatch @ self.weight_vh.T)
            assert (probs >= 0).all() and (probs <= 1).all()
            activations = sample_binary(probs) 
        
        return probs, activations
        #return np.zeros((n_samples,self.ndim_visible)), np.zeros((n_samples,self.ndim_visible))


    
    """ rbm as a belief layer : the functions below do not have to be changed until running a deep belief net """

    

    def untwine_weights(self):
        
        self.weight_v_to_h = np.copy( self.weight_vh )
        self.weight_h_to_v = np.copy( np.transpose(self.weight_vh) )
        self.weight_vh = None

    def get_h_given_v_dir(self,visible_minibatch):

        """Compute probabilities p(h|v) and activations h ~ p(h|v)

        Uses directed weight "weight_v_to_h" and bias "bias_h"
        
        Args: 
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:        
           tuple ( p(h|v) , h) 
           both are shaped (size of mini-batch, size of hidden layer)
        """
        
        assert self.weight_v_to_h is not None

        n_samples = visible_minibatch.shape[0]

        # [TODO TASK 4.2] perform same computation as the function 'get_h_given_v' but with directed connections (replace the zeros below) 
        probs = sigmoid(self.bias_h + visible_minibatch @ self.weight_v_to_h )
        assert (probs >= 0).all() and (probs <= 1).all()
        activations = sample_binary(probs) 
        return probs, activations 
        #return np.zeros((n_samples,self.ndim_hidden)), np.zeros((n_samples,self.ndim_hidden))


    def get_v_given_h_dir(self,hidden_minibatch):


        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses directed weight "weight_h_to_v" and bias "bias_v"
        
        Args: 
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:        
           tuple ( p(v|h) , v) 
           both are shaped (size of mini-batch, size of visible layer)
        """
        
        assert self.weight_h_to_v is not None
        
        n_samples = hidden_minibatch.shape[0]
        
        if self.is_top:

            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \ 
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """
            
            # [TODO TASK 4.2] Note that even though this function performs same computation as 'get_v_given_h' but with directed connections,
            # this case should never be executed : when the RBM is a part of a DBN and is at the top, it will have not have directed connections.
            # Appropriate code here is to raise an error (replace pass below)
            support = self.bias_v + hidden_minibatch @ self.weight_h_to_v.T
            support_vis = support[:, self.n_labels:]
            support_labels = support[:, 0:self.n_labels]

            probs_vis = sigmoid(support_vis)
            probs_labels = softmax(support_labels)
            probs = np.concatenate((probs_labels, probs_vis),axis=1)
            assert (probs >= 0).all() and (probs <= 1).all()

            act_vis = sample_binary(probs_vis) 
            act_labels = sample_categorical(probs_labels) 
            activations = np.concatenate((act_labels, act_vis),axis=1)

        else:
                        
            # [TODO TASK 4.2] performs same computaton as the function 'get_v_given_h' but with directed connections (replace the pass and zeros below)             

            probs = sigmoid(self.bias_v + hidden_minibatch @ self.weight_h_to_v)
            assert (probs >= 0).all() and (probs <= 1).all()
            activations = sample_binary(probs) 

        return probs, activations 

    def update_generate_params(self,inps,trgs,preds):
        
        """Update generative weight "weight_h_to_v" and bias "bias_v"
        
        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.3] find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.
        
        self.delta_weight_h_to_v += 0
        self.delta_bias_v += 0
        
        self.weight_h_to_v += self.delta_weight_h_to_v
        self.bias_v += self.delta_bias_v 
        
        return
    
    def update_recognize_params(self,inps,trgs,preds):
        
        """Update recognition weight "weight_v_to_h" and bias "bias_h"
        
        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.3] find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.

        self.delta_weight_v_to_h += 0
        self.delta_bias_h += 0

        self.weight_v_to_h += self.delta_weight_v_to_h
        self.bias_h += self.delta_bias_h
        
        return    

def sig(x):
    return 1. / (1. + np.exp(-x))

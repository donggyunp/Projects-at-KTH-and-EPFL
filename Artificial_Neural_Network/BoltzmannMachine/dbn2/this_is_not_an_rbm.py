from util import *
import numpy as np


class RestrictedBoltzmannMachine():
    def __init__(self, ndim_visible, ndim_hidden, is_bottom=False, image_size=[28, 28], is_top=False, n_labels=10,
                 batch_size=10):
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

        if is_bottom: self.image_size = image_size

        self.is_top = is_top

        if is_top: self.n_labels = 10

        self.batch_size = batch_size

        self.delta_bias_v = 0

        self.delta_weight_vh = 0

        self.delta_bias_h = 0

        self.bias_v = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible))  # b

        self.weight_vh = np.random.normal(loc=0.0, scale=0.01,
                                          size=(self.ndim_visible, self.ndim_hidden))  # nv x nh : W

        self.bias_h = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_hidden))  # g

        self.delta_weight_v_to_h = 0

        self.delta_weight_h_to_v = 0

        self.weight_v_to_h = None

        self.weight_h_to_v = None

        self.learning_rate = 0.01

        self.momentum = 0.7

        self.print_period = 5000

        self.rf = {  # receptive-fields. Only applicable when visible layer is input data
            "period": 5000,  # iteration period to visualize
            "grid": [5, 5],  # size of the grid
            "ids": np.random.randint(0, self.ndim_hidden, 25)  # pick some random hidden units
        }

        return



    def cd1(self, visible_trainset, labels=None, n_iterations=10000, total_iter=0, plotError=False, name=''):
        """Contrastive Divergence with k=1 full alternating Gibbs sampling
        Args:
          visible_trainset: training data for this rbm, shape is (size of training set, size of visible layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        iter_list = []
        if self.is_top:
            visible_trainset = np.concatenate((visible_trainset, labels), axis=1)

        print(total_iter)
        for it in range(n_iterations):

            minibatch = visible_trainset[self.batch_size * it:it * self.batch_size + self.batch_size]

            p_h0, h0 = self.get_h_given_v(minibatch)  # v_0 -> h_0, both are shaped 20x500

            p_v1, v1 = self.get_v_given_h(h0)  # h_0 -> v_1

            p_h1, h1 = self.get_h_given_v(p_v1)  # both are shaped 20x500

            total_iter += 1

            # [TODO TASK 4.1] update the parameters using function 'update_params'
            self.update_params(minibatch, p_h0, p_v1, p_h1)  # sending in <v0 h0>, <vk hk>, where first is <p, binary> and latter is <p, p>

        if self.is_top:
            return total_iter

        p_h0, h0 = self.get_h_given_v(visible_trainset) # v_0 -> h_0,

        p_v1, v1 = self.get_v_given_h(h0) # h_0 -> v_1

        MSE = np.sum(np.square(visible_trainset - p_v1))/visible_trainset.shape[0]

        return MSE, total_iter



    def update_params(self, v_0, h_0, v_k, h_k):
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
        # v_0 (samples), pv_k: 20x784
        # ph_0, ph_k: 20x500
        # W: 784x500
        # 784x500           # 784x500

        self.delta_bias_v = self.learning_rate * np.sum((v_0 - v_k), axis=0)
        self.delta_weight_vh = self.learning_rate * (np.matmul(v_0.T, h_0) - np.matmul(v_k.T, h_k))
        self.delta_bias_h = self.learning_rate * np.sum((h_0 - h_k), axis=0)

        self.bias_v += self.delta_bias_v
        self.weight_vh += self.delta_weight_vh
        self.bias_h += self.delta_bias_h



    def get_h_given_v(self, visible_minibatch):
        """Compute probabilities p(h|v) and activations h ~ p(h|v)

        Uses undirected weight "weight_vh" and bias "bias_h"

        Args:
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:
           tuple ( p(h|v) , h)
           both are shaped (size of mini-batch, size of hidden layer)
        """

        assert self.weight_vh is not None

        # W: vn x hn
        # hn: 500, vn: 784          |     hn: 2000,  vn: 510
        # vis_minibatch: 20x784
        # bias_h: 500

        # h = 20x500
        h = np.matmul(visible_minibatch, self.weight_vh) + self.bias_h  # each row in h is the y for one datapoint

        p_h = sigmoid(h)

        activations = sample_binary(p_h)

        return p_h, activations



    def get_v_given_h(self, hidden_minibatch):
        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses undirected weight "weight_vh" and bias "bias_v"

        Args:
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:
           tuple ( p(v|h) , v)
           both are shaped (size of mini-batch, size of visible layer)
        """

        assert self.weight_vh is not None

        if self.is_top:
            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """
            # [TODO TASK 4.1] compute probabilities and activations (samples from probabilities) of visible layer (replace the pass below). \
            # Note that this section can also be postponed until TASK 4.2, since in this task, stand-alone RBMs do not contain labels in visible layer.

            support = np.matmul(hidden_minibatch, self.weight_vh.T) + self.bias_v

            data_support, label_support = support[:, :-self.n_labels], support[:, -self.n_labels:] # split into label and data

            # logistic for data, and softmax for labels
            p_label = softmax(label_support)
            p_data = sigmoid(data_support)
            p_v = np.concatenate((p_data, p_label), axis=1)
            a_v = np.concatenate((sample_binary(p_data), sample_categorical(p_label)), axis=1) # TRY DOING SAMPLE CATEGORICAL ON p_label!!! ------------ OBS

        else:
            v = np.matmul(hidden_minibatch, self.weight_vh.T) + self.bias_v

            p_v = sigmoid(v)

            a_v = sample_binary(p_v)

        return p_v, a_v





    """ rbm as a belief layer : the functions below do not have to be changed until running a deep belief net """



    def untwine_weights(self):
        self.weight_v_to_h = np.copy(self.weight_vh)
        self.weight_h_to_v = np.copy(np.transpose(self.weight_vh))
        self.weight_vh = None



    def get_h_given_v_dir(self, visible_minibatch):
        """Compute probabilities p(h|v) and activations h ~ p(h|v)
        Uses directed weight "weight_v_to_h" and bias "bias_h"
        Args:
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:
           tuple ( p(h|v) , h)
           both are shaped (size of mini-batch, size of hidden layer)
        """
        assert self.weight_v_to_h is not None
        v = np.matmul(visible_minibatch, self.weight_v_to_h) + self.bias_h
        p_v = sigmoid(v)
        a_v = sample_binary(p_v)

        return p_v, a_v



    def get_v_given_h_dir(self, hidden_minibatch):
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
            print("ERROR - when the RBM is a part of a DBN and is at the top, it will have not have directed connections")


        else:

            # [TODO TASK 4.2] performs same computaton as the function 'get_v_given_h' but with directed connections (replace the pass and zeros below)

            v = np.matmul(hidden_minibatch, self.weight_h_to_v) + self.bias_v

            p_v = sigmoid(v)

            a_v = sample_binary(p_v)

        #return np.zeros((n_samples, self.ndim_visible)), np.zeros((n_samples, self.ndim_visible))
        return p_v, a_v



    def update_generate_params(self, inps, trgs, preds):
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



    def update_recognize_params(self, inps, trgs, preds):
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

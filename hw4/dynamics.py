import tensorflow as tf
import numpy as np
import math

# Predefined function to build a feedforward neural network
def build_mlp(input_placeholder, 
              output_size,
              scope, 
              n_layers=2, 
              size=500, 
              activation=tf.tanh,
              output_activation=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out

class NNDynamicsModel():
    def __init__(self, 
                 env, 
                 n_layers,
                 size, 
                 activation, 
                 output_activation, 
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess
                 ):
        """ YOUR CODE HERE """
        """ Note: Be careful about normalization """
        self.epsilon = 1e-7
        self.env = env
        self.n_layers = n_layers
        self.size = size
        self.activation = activation
        self.output_activation = output_activation
        self.mean_obs, self.std_obs, self.mean_deltas, self.std_deltas, self.mean_action, self.std_action = normalization
        self.batch_size = batch_size
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.sess = sess
        
        self.ob_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.shape[0]

        # set the placeholders and tensorflow graphs
        self.st_at =tf.placeholder(shape=[None, self.ob_dim+self.ac_dim], name="ob", dtype=tf.float32)
        self.delta = build_mlp(input_placeholder = self.st_at, output_size = self.ob_dim, scope = 'dyn', n_layers = n_layers, size = size, activation = activation, output_activation = output_activation)

        self.delta_ =tf.placeholder(shape=[None, self.ob_dim], name="stp1_", dtype=tf.float32)

        # compute the loss and initialize the optimizer
        self.nrm = tf.nn.l2_loss(self.delta- self.delta_)
        self.update_op = tf.train.AdamOptimizer(learning_rate).minimize(self.nrm)

    def fit(self, data):
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """

        """YOUR CODE HERE """
        # unormalized data
        un_st = np.concatenate([datum["observations"] for datum in data])
        un_stp1 = np.concatenate([datum["next_observations"] for datum in data])
        un_at = np.concatenate([datum["actions"] for datum in data])
        
        # normalize data
        n_st = (un_st-self.mean_obs)/(self.std_obs+self.epsilon)
        n_at = (un_at-self.mean_action)/(self.std_action+self.epsilon)
        n_stat = np.concatenate([n_st,n_at],axis=1)
        
        n_delta = ((un_stp1-un_st)-self.mean_deltas)/(self.std_deltas+self.epsilon)

        # make a shuffle row of whole data to be used
        N = n_delta.shape[0]
        train_indicies = np.arange(N)
        np.random.shuffle(train_indicies)
        # train over the whole data set for the number of iterations
        for i in range(self.iterations):
            for i in range(int(math.ceil(N/self.batch_size))):
                # index for the batch points from a random row
                start_idx = i*self.batch_size%N
                idx = train_indicies[start_idx:start_idx+self.batch_size]
                # choose the batch
                feed_dict = {self.st_at : n_stat[idx,:], self.delta_ : n_delta[idx,:]}
                # train the data
                self.sess.run(self.update_op, feed_dict=feed_dict)

    def predict(self, states, actions):
        """ Write a function to take in a batch of (unnormalized) states and (unnormalized) actions and return the (unnormalized) next states as predicted by using the model """
        """ YOUR CODE HERE """
        # normalize the data
        n_states = (states-self.mean_obs)/(self.std_obs+self.epsilon)
        n_actions = (actions-self.mean_action)/(self.std_action+self.epsilon)
        n_stat = np.concatenate([n_states,n_actions],axis=1)
        
        # predict using the model and unnromalize
        feed_dict = {self.st_at : n_stat}
        n_stp1 = self.sess.run(self.delta, feed_dict=feed_dict)

        un_stp1 = n_stp1*self.std_deltas + self.mean_deltas + states

        return un_stp1










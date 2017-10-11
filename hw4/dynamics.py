import tensorflow as tf
import numpy as np

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

        self.st_at =tf.placeholder(shape=[None, ob_dim+ac_dim], name="ob", dtype=tf.float32)
        self.delta = build_mlp(input_placeholder = st_at, ob_dim, scope = sess, n_layers = n_layers, size = size, activation = activation, output_activation = output_activation)

        self.delta_ =tf.placeholder(shape=[None, ob_dim], name="stp1_", dtype=tf.float32)

        self.nrm = tf.nn.l2_loss(delta-delta_)
        self.update_op = tf.train.AdamOptimizer(learning_rate).minimize(nrm)

    def fit(self, data):
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """

        """YOUR CODE HERE """
        un_st = data["observations"]
        un_stp1 = data["next_observations"]
        un_at = data["actions"]
        
        n_st = (un_st-mean_obs)/(std_obs+epsilon)
        n_at = (un_at-mean_action)/(std_action+epsilon)
        n_stat = np.concatenate(n_st,n_at,axis=1)

        n_delta = ((un_stp1-unst)-mean_deltas)/(std_deltas+epsilon)

        feed_dict = {st_at : n_stat, delta_ : n_delta}

        for i in range(iterations):
            sess.run(update_op, feed_dict=feed_dict)

    def predict(self, states, actions):
        """ Write a function to take in a batch of (unnormalized) states and (unnormalized) actions and return the (unnormalized) next states as predicted by using the model """
        """ YOUR CODE HERE """
        
        n_states = (states-mean_obs)/(std_obs+epsilon)
        n_actions = (un_at-mean_action)/(std_action+epsilon)
        n_stat = np.concatenate(n_states,n_actions,axis=1)
        
        feed_dict = {st_at : n_stat}
        n_sp1 = sess.run(delta, feed_dict=feed_dict)

        un_stp1 = n_stp1*std_delta + mean_deltas + actions

        return un_stp1










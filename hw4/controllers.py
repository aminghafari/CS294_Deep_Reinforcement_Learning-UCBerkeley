import numpy as np
from cost_functions import trajectory_cost_fn
import time

class Controller():
	def __init__(self):
		pass

	# Get the appropriate action(s) for this state(s)
	def get_action(self, state):
		pass


class RandomController(Controller):
	def __init__(self, env):
		""" YOUR CODE HERE """
		self.env = env

	def get_action(self, state):
		""" YOUR CODE HERE """
		""" Your code should randomly sample an action uniformly from the action space """
		return self.env.action_space.sample()


class MPCcontroller(Controller):
	""" Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """
	def __init__(self, 
				 env, 
				 dyn_model, 
				 horizon=5, 
				 cost_fn=None, 
				 num_simulated_paths=10,
				 ):
		self.env = env
		self.dyn_model = dyn_model
		self.horizon = horizon
		self.cost_fn = cost_fn
		self.num_simulated_paths = num_simulated_paths

	def get_action(self, state):
            """ YOUR CODE HERE """
            """ Note: be careful to batch your simulations through the model for speed """
            # to store the cost and the first action of each path
            cost = []
            first_action = []
            
            # genrate the path
            # first set of states
            st = np.repeat([state], self.num_simulated_paths, axis = 0)
            cost = np.zeros((self.num_simulated_paths,))
            # loop for hirzon
            for n_hz in range(self.horizon):
                ac = []
                # genrate random actions
                for n_path in range(self.num_simulated_paths):
                    ac.append(self.env.action_space.sample())
                    if(n_hz==0):
                        ac_first = np.array(ac)
                # predict using dynamic model
                stp1 = self.dyn_model.predict(st, np.array(ac))
                
                # find the cost
                cost = cost + self.cost_fn(st, np.array(ac), stp1)
                st = stp1

            # find the low cost path
            arg_min = np.argmin(cost)
            
            # choose and return the action
            return ac_first[arg_min,:]



















                    

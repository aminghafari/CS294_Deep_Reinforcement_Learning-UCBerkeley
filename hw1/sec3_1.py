import numpy as np
import pickle
import tensorflow as tf
import tf_util
import load_policy
import matplotlib.pyplot as plt
import math
import model

def main():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('expert_policy_file', type=str)
	parser.add_argument('expert_data',type=str)
	parser.add_argument('envname', type=str)
	parser.add_argument('--render', action='store_true')
	parser.add_argument('--epochs', type=int, default=20)
	parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
	args = parser.parse_args()

        # load the data
	expert_data = pickle.load( open( args.expert_data, "rb" ) )
        
	# Training data
	x_train = expert_data['observations']
	y_train = expert_data['actions']
	y_train = np.squeeze(y_train, axis=1)


	# make the environmet
	import gym
	env = gym.make(args.envname)
	
	# Epochs
	returns_mean = []
	returns_std = []
	with tf.Session() as sess:
		# train the model
		losses = model.train_model(x_train, y_train, sess, args.epochs)
		
		# Get the observations
		observations, returns = model.run_trained_model(sess, env, args.num_rollouts, args.render)
		observations = np.array(observations)
	tf.reset_default_graph()

	print('mean return:', np.mean(returns))
	print('std of return', np.std(returns))


if __name__ == '__main__':
    main()

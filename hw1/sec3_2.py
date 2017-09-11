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
	parser.add_argument('--epoch_step', type=int, default=5)
	parser.add_argument('--epochs_itr', type=int, default=10)
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
	
	# Train for different of number of epochs
	returns_mean = []
	returns_std = []
	epoch_vec = []
	epoch = 0
	for i in range(args.epochs_itr):
		print('epochs iteration number:', i+1, '/', args.epochs_itr)

		with tf.Session() as sess:
			# train the model
			epoch += args.epoch_step
			epoch_vec.append(epoch)
			losses = model.train_model(x_train, y_train, sess, epoch)
			
			# Get the observations
			observations, returns = model.run_trained_model(sess, env, args.num_rollouts, args.render)
			observations = np.array(observations)
			returns_mean.append(np.mean(returns))
			returns_std.append(np.std(returns))
		tf.reset_default_graph()


	fig = plt.figure()
	plt.plot(epoch_vec, returns_mean)
	plt.errorbar(epoch_vec, returns_mean, returns_std, linestyle='None', marker='^')
	plt.ylabel('The trained policy’s mean return')
	plt.xlabel('Number of epochs iterations for ' + args.envname)
	plt.show()
	fig.savefig('Epochs-'+args.envname+'.png')


if __name__ == '__main__':
    main()

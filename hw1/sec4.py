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
	parser.add_argument('--dagger_itr', type=int, default=10)
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
	
	# DAgger Algorithm
	returns_mean = []
	returns_std = []

	# returns without DAgger 
	with tf.Session() as sess:
		# train the model
		losses = model.train_model(x_train, y_train, sess, args.epochs)
		
		# Get the observations from the grained model
		observations, returns = model.run_trained_model(sess, env, args.num_rollouts, args.render)
		observations = np.array(observations)
		returns_mean.append(np.mean(returns))
		returns_std.append(np.std(returns))
		r_mean_BC = np.mean(returns)
		r_std_BC = np.mean(returns)
	tf.reset_default_graph()
				
		
	for i in range(args.dagger_itr):
		print('DAgger Itertions:', i+1, '/', args.dagger_itr)
		actions = []
		

		# Get the actions from expert policy
		print('loading and building expert policy')
		policy_fn = load_policy.load_policy(args.expert_policy_file)
		print('loaded and built')

		
		with tf.Session() as sess1:
			tf_util.initialize()
			N_obs = observations.shape[0]
			for j in range(N_obs):
				obs = observations[j,:]
				action = policy_fn(obs[None,:])
				actions.append(action)
			
			actions = np.squeeze(np.array(actions), axis=1)
			x_train = np.append(x_train, observations, axis = 0)
			y_train = np.append(y_train, actions, axis = 0)
			#tf.reset_default_graph()
		tf.reset_default_graph()
	
		with tf.Session() as sess:
			# train the model
			losses = model.train_model(x_train, y_train, sess, args.epochs)
			
			# Get the observations from the trained model
			observations, returns = model.run_trained_model(sess, env, args.num_rollouts, args.render)
			observations = np.array(observations)
			returns_mean.append(np.mean(returns))
			returns_std.append(np.std(returns))
		tf.reset_default_graph()

	## Plotting the results
	# mean and std for the expret policy and Behavioral Cloning
	retunrs_expert = expert_data['returns']
	r_mean_ex = np.mean(retunrs_expert)
	r_std_ex = np.std(retunrs_expert)
	r_mean_BC = r_mean_BC
	r_std_BC = r_std_BC

	# fig
	fig = plt.figure()
	plt.errorbar(np.arange(args.dagger_itr+1), returns_mean, returns_std, marker='^',fmt="b")
	l2, = plt.plot([0,args.dagger_itr], [r_mean_ex,r_mean_ex],"g--", label="Expert Ploicy Mean")
	l3, = plt.plot([0,args.dagger_itr], [r_mean_BC,r_mean_BC],"r--", label="Behavioral Learning Mean")
	plt.legend(handles=[l2,l3], loc=2)
	plt.ylabel('The trained policy’s mean return')
	plt.xlabel('Number of DAgger iterations for ' + args.envname)
	plt.show()
	fig.savefig('DAgger-'+args.envname+'.png')


if __name__ == '__main__':
    main()

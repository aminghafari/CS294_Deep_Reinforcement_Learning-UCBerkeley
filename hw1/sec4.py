import numpy as np
import pickle
import tensorflow as tf
import tf_util
import load_policy
import matplotlib.pyplot as plt
import math

def model(x,out_D):
	# setup variables
	hD1 = 40
	hD2 = 40
	hD3 = 20
	# layer1
	W1 = tf.get_variable("W1", shape = [x.shape[1], hD1])
	b1 = tf.get_variable("b1", shape = [1, hD1])
	# layer2
	W2 = tf.get_variable('W2', shape = [hD1, hD2])
	b2 = tf.get_variable('b2', shape = [1, hD2])
	# layer3
	W3 = tf.get_variable('W3', shape = [hD2, hD3])
	b3 = tf.get_variable('b3', shape = [1, hD3])
	# layer4
	W4 = tf.get_variable('W4', shape = [hD3, out_D])
	b4 = tf.get_variable('b4', shape = [1, out_D])
	# graph
	h1 = tf.nn.relu( tf.matmul(x,  W1) + b1 )
	h2 = tf.nn.relu( tf.matmul(h1, W2) + b2 )
	h3 = tf.nn.relu( tf.matmul(h2, W3) + b3 )
	y_out = tf.add(tf.matmul(h3, W4), b4, name="y_out")

	return y_out


def train_model(x_train, y_train, sess, epochs):
	# get the number of data
	N = x_train.shape[0]
	# make a shuffled version of order of the data
	train_indicies = np.arange(N)
	np.random.shuffle(train_indicies)
	# output dimension
	out_D = y_train.shape[1]

	# Model
	x = tf.placeholder(tf.float32, [None, x_train.shape[1]], name = "x")
	y = tf.placeholder(tf.float32, [None, y_train.shape[1]])
	is_training = tf.placeholder(tf.bool)
	# output of the model
	y_out = model(x, out_D)
	# loss
	loss = tf.reduce_mean(tf.square(y_out - y))
	# optimizer
	optimizer = tf.train.AdamOptimizer(5e-4)
	train = optimizer.minimize(loss)
	
	# initialize the variables
	sess.run(tf.global_variables_initializer())

	batch_size = 100
	losses = []
	for epoch in range(epochs):
		for i in range(int(math.ceil(N/batch_size))):
			start_idx = i*batch_size%N
			idx = train_indicies[start_idx:start_idx+batch_size]
			# create a feed dictionary
			feed_dict = {x : x_train[idx,:], y: y_train[idx,:], is_training: True}
			
			loss_step,_ = sess.run([loss, train], feed_dict=feed_dict)
			
			losses.append(loss_step)
		print('epoch:',epoch,', mean_loss:',loss_step)
	return losses

def run_trained_model(sess, env, num_rollouts, render):
	# run the trained policy on data
	returns = []
	observations = []
	actions = []
	
	# retrieve the placeholders
	graph = tf.get_default_graph()
	x = graph.get_tensor_by_name("x:0")
	y_out = graph.get_tensor_by_name("y_out:0")
	
	max_steps = env.spec.timestep_limit
	for i in range(num_rollouts):
		print('iter', i)
		obs = env.reset()
		done = False
		totalr = 0.
		steps = 0
		while not done:
			action = sess.run(y_out, feed_dict={x : np.array([obs])})
			observations.append(obs)
			actions.append(action)
			obs, r, done, _ = env.step(action)
			totalr += r
			steps += 1
			
			if render:
				env.render()
			if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
			if steps >= max_steps:
				break
		returns.append(totalr)

	return observations, returns

def main():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('expert_policy_file', type=str)
	parser.add_argument('expert_data',type=str)
	parser.add_argument('envname', type=str)
	parser.add_argument('--render', action='store_true')
	parser.add_argument('--epochs', type=int, default=20)
	parser.add_argument('--dagger_itr', type=int, default=5)
	parser.add_argument('--num_rollouts', type=int, default=2,
                        help='Number of expert roll outs')
	args = parser.parse_args()

        # load the data
	expert_data = pickle.load( open( args.expert_data, "rb" ) )
        
	# Training data
	x_train = expert_data['observations']
	y_train = expert_data['actions']
	y_train = np.squeeze(y_train, axis=1)

	# load the expert policy
	print('loading and building expert policy')
	policy_fn = load_policy.load_policy(args.expert_policy_file)
	print('loaded and built')

	# make the environmet
	import gym
	env = gym.make(args.envname)
	
	# DAgger Algorithm
	for i in range(args.dagger_itr):
		print('DAgger Itertions:', i, 'out of', args.dagger_itr)
		actions = []
		# train the model
		# make the tensorflow session
		sess = tf.Session()
		losses = train_model(x_train, y_train, sess, args.epochs)
		# Get the observations
		observations, returns = run_trained_model(sess, env, args.num_rollouts, args.render)
		sess.close()
		# Get the actions from expert policy
		with tf.Session():
			tf_util.initialize()
			N_obs = np.array(observations).shape[0]
			for j in range(N_obs):
				obs = np.array(observations)[j,:]
				action = policy_fn(obs[None,:])
				actions.append(action)
			
			print(x_train.shape)
			print(np.array(observations).shape)
			x_train.append(observations)
			y_train.append(actions)



if __name__ == '__main__':
    main()

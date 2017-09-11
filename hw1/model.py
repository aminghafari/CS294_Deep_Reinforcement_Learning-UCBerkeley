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
	graph = sess.graph
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

	#tf.Session.reset()
	return observations, returns

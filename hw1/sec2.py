import numpy as np
import pickle
import tensorflow as tf
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


def main():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('expert_data',type=str)
	parser.add_argument('envname', type=str)
	parser.add_argument('--epochs', type=int, default=20)
	args = parser.parse_args()

        # load the data
	expert_data = pickle.load( open( args.expert_data, "rb" ) )
        
	# Training data
	x_train = expert_data['observations']
	y_train = expert_data['actions']
	y_train = np.squeeze(y_train, axis=1)

	# make the tensorflow session
	sess = tf.Session()
	
	# train the model
	losses = train_model(x_train, y_train, sess, args.epochs)

	saver = tf.train.Saver()
	saver.save(sess, args.envname+"-trained")

	plt.plot(losses)
	plt.ylabel('loss')
	plt.xlabel('iterations of batch size = 100, 5 epochs, data size = 20000 ')
	plt.show()


if __name__ == '__main__':
    main()

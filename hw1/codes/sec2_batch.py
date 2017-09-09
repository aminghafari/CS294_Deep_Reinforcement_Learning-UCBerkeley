import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import math

def model(x,y):
	# setup variables
	hD1 = 40
	hD2 = 40
	hD3 = 3
	# layer1
	W1 = tf.get_variable("W1", shape = [x.shape[1], hD1])
	b1 = tf.get_variable("b1", shape = [1, hD1])
	# layer2
	W2 = tf.get_variable('W2', shape = [hD1, hD2])
	b2 = tf.get_variable('b2', shape = [1, hD2])
	# layer3
	W3 = tf.get_variable('W3', shape = [hD2, hD3])
	b3 = tf.get_variable('b3', shape = [1, hD3])
	# graph
	h1 = tf.nn.elu( tf.matmul(x, W1) + b1 )
	h2 = tf.nn.elu( tf.matmul(h1, W2) + b2 )
	y_out = tf.matmul(h2, W3) + b3

	return y_out


def main():
        # load the data
	expert_data = pickle.load( open( "expert_data.pkl", "rb" ) )
        
	# Training data
	x_train = expert_data['observations']
	y_train = expert_data['actions']
	y_train = np.squeeze(y_train, axis=1)

	N = x_train.shape[0]
	print(int(math.ceil(N/100)))
	train_indicies = np.arange(N)
	np.random.shuffle(train_indicies)

	# Model
	x = tf.placeholder(tf.float32, [None, x_train.shape[1]])
	y = tf.placeholder(tf.float32, [None, y_train.shape[1]])
	is_training = tf.placeholder(tf.bool)
	# output of the model
	y_out = model(x, y)
	# loss
	loss = tf.reduce_mean(tf.square(y_out - y))
	# optimizer
	optimizer = tf.train.AdamOptimizer(5e-4)
	train = optimizer.minimize(loss)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	
	batch_size = 100
	losses = []
	for epoch in range(5):
		print('epoch:',epoch)
		for i in range(int(math.ceil(N/batch_size))):
			start_idx = i*batch_size%N
			idx = train_indicies[start_idx:start_idx+batch_size]
			# create a feed dictionary
			feed_dict = {x : x_train[idx,:], y: y_train[idx,:], is_training: True}
			# get the batch size			
			actual_batch_size = y_train[idx].shape[0]
			
			loss_step,_ = sess.run([loss, train], feed_dict=feed_dict)
			
			losses.append(loss_step)
			print(loss_step)

	plt.plot(losses)
	plt.ylabel('loss')
	plt.xlabel('iterations of batch size = 100, 5 epochs, data size = 20000 ')
	plt.show()

if __name__ == '__main__':
    main()

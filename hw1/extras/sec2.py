import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt

def model(x,y):
	# setup variables
	hD1 = 30
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

	# Model
	x = tf.placeholder(tf.float32, [None, x_train.shape[1]])
	y = tf.placeholder(tf.float32, [None, y_train.shape[1]])
	# output of the model
	y_out = model(x, y)
	# loss
	loss = tf.reduce_mean(tf.square(y_out - y_train))
	# optimizer
	optimizer = tf.train.AdamOptimizer(5e-4)
	train = optimizer.minimize(loss)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	
	loss_ = []
	for step in range(0,4001):
		train.run({x: x_train}, sess)
		#if step%100==0 :
		loss_step = loss.eval({x: x_train}, sess)
		print(loss_step)
		loss_.append(loss_step)

	plt.plot(loss_)
	plt.ylabel('loss')
	plt.xlabel('iterations')
	plt.show()

if __name__ == '__main__':
    main()

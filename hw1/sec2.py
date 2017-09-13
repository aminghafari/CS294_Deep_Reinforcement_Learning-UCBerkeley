import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import model


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
	losses = model.train_model(x_train, y_train, sess, args.epochs)

	#saver = tf.train.Saver()
	#saver.save(sess, args.envname+"-trained")

	plt.plot(losses)
	plt.ylabel('loss')
	plt.xlabel('iterations of batch size = 100, 5 epochs, data size = 20000 ')
	plt.show()


if __name__ == '__main__':
    main()

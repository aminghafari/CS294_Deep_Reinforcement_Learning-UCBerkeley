import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math


def main():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('trained_model',type=str)
	parser.add_argument('envname', type=str)
	parser.add_argument('--render', action='store_true')
	parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
	args = parser.parse_args()


	# retrieve the session
	with tf.Session() as sess:
		new_saver = tf.train.import_meta_graph(args.trained_model)
		new_saver.restore(sess, tf.train.latest_checkpoint('./'))

		# import the graph and the output
		graph = tf.get_default_graph()
		x = graph.get_tensor_by_name("x:0")
		y_out = graph.get_tensor_by_name("y_out:0")
	
		# run the trained policy on data
		import gym
		env = gym.make(args.envname)
		max_steps = env.spec.timestep_limit
		print(max_steps)

		returns = []
		observations = []
		actions = []
		for i in range(args.num_rollouts):
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
				
				if args.render:
					env.render()
				if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
				if steps >= max_steps:
					break
			returns.append(totalr)
	
		#returns_expert = expert_data['returns']
		print('mean return:', np.mean(returns))
		#print('mean return for expert:', np.mean(returns_expert))
		print('std of return', np.std(returns))
		#print('std of return for expert', np.std(returns_expert))
	
if __name__ == '__main__':
    main()

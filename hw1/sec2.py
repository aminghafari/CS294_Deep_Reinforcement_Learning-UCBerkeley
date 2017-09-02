import numpy as np
import pickle
import tensorflow as tf



def main():
	expert_data = pickle.load( open( "expert_data.pkl", "rb" ) )

	print(expert_data['observations'].shape)
	print(expert_data['actions'].shape)


if __name__ == '__main__':
    main()

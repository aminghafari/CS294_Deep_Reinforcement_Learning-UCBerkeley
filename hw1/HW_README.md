# Amin Ghafari Zeydabadi 25911193,  11 September 2017, HW1, Deep Reinforcement Learning

To get the equivalent results to the ones reported in the report please refer to the following codes:


# Section 2
Run : python run_expert.py experts/Hopper-v1.pkl Hopper-v1 --num_rollouts 20
and then
Run : python sec2.py  Hopper-v1-data.pkl Hopper-v1 --epochs 20

# Section 3.1
Run : python run_expert.py experts/Hopper-v1.pkl Hopper-v1 --num_rollouts 20
and then
Run : python sec3_1.py experts/Hopper-v1.pkl Hopper-v1-data.pkl Hopper-v1 --epochs 20 --num_rollouts 20 --render

and 

Run : python run_expert.py experts/Ant-v1.pkl Ant-v1 --num_rollouts 20
and then
Run : python sec3_1.py experts/Ant-v1.pkl Ant-v1-data.pkl Ant-v1 --epochs 20 --num_rollouts 20 --render

# Section 3.2
Run : python run_expert.py experts/Hopper-v1.pkl Hopper-v1 --num_rollouts 20
and then
Run : python sec3_2.py experts/Hopper-v1.pkl Hopper-v1-data.pkl Hopper-v1 --epoch_step 20 --num_rollouts 20 --epochs_itr 10 --render

and

Run : python run_expert.py experts/Ant-v1.pkl Ant-v1 --num_rollouts 20
and then
Run : python sec3_2.py experts/Ant-v1.pkl Ant-v1-data.pkl Ant-v1 --epoch_step 20 --num_rollouts 20 --epochs_itr 10 --render



# Section 4
Run : python run_expert.py experts/Walker2d-v1.pkl Walker2d-v1 --num_rollouts 20
and then
Run : python sec4.py experts/Hopper-v1.pkl Hopper-v1-data.pkl Hopper-v1 --epochs 20 --num_rollouts 20 --dagger_itr 10 --render


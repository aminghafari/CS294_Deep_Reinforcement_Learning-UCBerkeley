import multiprocess
from time import clock as time
from time import sleep

import numpy as np
import tensorflow as tf
from parallel_trpo.utils import filter_ob, make_network

from rl_teacher.segment_sampling import sample_segment_from_path

class Actor(multiprocess.Process):
    def __init__(self, task_q, result_q, env_id, make_env, seed, max_timesteps_per_episode):
        multiprocess.Process.__init__(self)
        self.env_id = env_id
        self.make_env = make_env
        self.seed = seed
        self.task_q = task_q
        self.result_q = result_q

        self.max_timesteps_per_episode = max_timesteps_per_episode

    # TODO Cleanup
    def set_policy(self, weights):
        placeholders = {}
        assigns = []
        for var in self.policy_vars:
            placeholders[var.name] = tf.placeholder(tf.float32, var.get_shape())
            assigns.append(tf.assign(var, placeholders[var.name]))

        feed_dict = {}
        count = 0
        for var in self.policy_vars:
            feed_dict[placeholders[var.name]] = weights[count]
            count += 1
        self.session.run(assigns, feed_dict)

    def act(self, obs):
        obs = np.expand_dims(obs, 0)
        avg_action_dist, logstd_action_dist = self.session.run(
            [self.avg_action_dist, self.logstd_action_dist], feed_dict={self.obs: obs})
        # samples the guassian distribution
        act = avg_action_dist + np.exp(logstd_action_dist) * np.random.randn(*logstd_action_dist.shape)
        return act.ravel(), avg_action_dist, logstd_action_dist

    def run(self):
        self.env = self.make_env(self.env_id)
        self.env.seed = self.seed

        # tensorflow variables (same as in model.py)
        observation_size = self.env.observation_space.shape[0]
        hidden_size = 64
        action_size = np.prod(self.env.action_space.shape)

        # tensorflow model of the policy
        self.obs = tf.placeholder(tf.float32, [None, observation_size])

        self.policy_vars, self.avg_action_dist, self.logstd_action_dist = make_network(
            "policy-a", self.obs, hidden_size, action_size)

        config = tf.ConfigProto(
            device_count = {'GPU': 0}
        )
        self.session = tf.Session(config=config)
        self.session.run(tf.global_variables_initializer())

        while True:
            next_task = self.task_q.get(block=True)
            if next_task == "do_rollout":
                # the task is an actor request to collect experience
                path = self.rollout()
                self.task_q.task_done()
                self.result_q.put(path)
            elif next_task == "kill":
                print("Received kill message. Shutting down...")
                self.task_q.task_done()
                break
            else:
                # the task is to set parameters of the actor policy
                self.set_policy(next_task)

                # super hacky method to make sure when we fill the queue with set parameter tasks,
                # an actor doesn't finish updating before the other actors can accept their own tasks.
                sleep(0.1)
                self.task_q.task_done()

    def rollout(self):
        obs, actions, rewards, avg_action_dists, logstd_action_dists, human_obs = [], [], [], [], [], []
        ob = filter_ob(self.env.reset())
        for i in range(self.max_timesteps_per_episode):
            action, avg_action_dist, logstd_action_dist = self.act(ob)

            obs.append(ob)
            actions.append(action)
            avg_action_dists.append(avg_action_dist)
            logstd_action_dists.append(logstd_action_dist)

            ob, rew, done, info = self.env.step(action)
            ob = filter_ob(ob)

            rewards.append(rew)
            human_obs.append(info.get("human_obs"))

            if done or i == self.max_timesteps_per_episode - 1:
                path = {
                    "obs": np.concatenate(np.expand_dims(obs, 0)),
                    "avg_action_dist": np.concatenate(avg_action_dists),
                    "logstd_action_dist": np.concatenate(logstd_action_dists),
                    "rewards": np.array(rewards),
                    "actions": np.array(actions),
                    "human_obs": np.array(human_obs)}
                return path

class ParallelRollout(object):
    def __init__(self, env_id, make_env, reward_predictor, num_workers, max_timesteps_per_episode, seed, num_r):
        self.num_workers = num_workers
        self.predictor = reward_predictor

        self.tasks_q = multiprocess.JoinableQueue()
        self.results_q = multiprocess.Queue()

        self.actors = []
        for i in range(self.num_workers):
            new_seed = seed * 1000 + i  # Give each actor a uniquely seeded env
            self.actors.append(Actor(self.tasks_q, self.results_q, env_id, make_env, new_seed, max_timesteps_per_episode))

        for a in self.actors:
            a.start()

        # we will start by running 20,000 / 1000 = 20 episodes for the first iteration  TODO OLD
        self.average_timesteps_in_episode = 1000

        # number of NN for reward function
        self.num_r = num_r

    def rollout(self, timesteps):
        start_time = time()
        # keep 20,000 timesteps per update  TODO OLD
        num_rollouts = int(timesteps / self.average_timesteps_in_episode)

        for _ in range(num_rollouts):
            self.tasks_q.put("do_rollout")
        self.tasks_q.join()

        paths = []
        paths_scores = np.zeros((num_rollouts))
        score_threshold = 0.5
        for i in range(num_rollouts):
            # bad_path = True
            # while bad_path:
            #     path = self.results_q.get()
                
            #     # get a segment from the path
            #     init_seg = sample_segment_from_path(path, int(self.predictor._frames_per_segment))

            #     paths_scores[i] = self.predictor.predict_segment_quality(init_seg)[1]

            #     print(self.results_q.qsize())
            #     if paths_scores[i]>= score_threshold:
            #         bad_path = False
            path = self.results_q.get()
            ################################
            #  START REWARD MODIFICATIONS  #
            ################################
            path["original_rewards"] = path["rewards"]
            path["rewards"] = self.predictor.predict_reward(path)
            self.predictor.path_callback(path)
            ################################
            #   END REWARD MODIFICATIONS   #
            ################################


            # get a segment from the path
            init_seg = sample_segment_from_path(path, int(self.predictor._frames_per_segment))

            paths_scores[i] = self.predictor.predict_segment_quality(init_seg)[1]


            paths.append(path)


        # # choose the best paths among the givens
        num_good_paths = 4
        idx_good_paths = paths_scores.argsort()[::-1][0:num_good_paths]

        paths = [paths[i] for i in idx_good_paths]



        
        # path selection
        # choose the paths that reward functions has the most agreement on them
        # paths_std   = np.zeros((num_rollouts))
        # ind_reward = np.zeros((self.num_r))
        # num_good_paths = 4
        # for i in range(num_rollouts):
            
        #     for j in range(self.num_r):
        #         ind_reward[j] = self.predictor.predict_segment_individual_reward(paths[i], j)

        #     paths_std[i] = np.std(ind_reward)

        # idx_good_paths = paths_std.argsort()[0:num_good_paths]

        # paths = [paths[i] for i in idx_good_paths]

        # path selection by the quality of segment




        self.average_timesteps_in_episode = sum([len(path["rewards"]) for path in paths]) / len(paths)

        return paths[0:4], time() - start_time

    def set_policy_weights(self, parameters):
        for i in range(self.num_workers):
            self.tasks_q.put(parameters)
        self.tasks_q.join()

    def end(self):
        for i in range(self.num_workers):
            self.tasks_q.put("kill")


### rollout with explorations
class ParallelRollout_1(object):
    def __init__(self, env_id, make_env, reward_predictor, num_workers, max_timesteps_per_episode, seed, num_r, num_policy):
        # to get paths from exploration policies
        self.num_policy = num_policy
        self.tasks_q_s = []
        self.results_q_s = []

        # time elpased
        self._timesteps_elapsed = 0


        self.num_workers = num_workers
        self.predictor = reward_predictor
        self.actors = []

        # a for loop for exploration policy
        for i in range(num_policy+1):
            tasks_q = multiprocess.JoinableQueue()
            results_q = multiprocess.Queue()

            self.tasks_q_s.append(tasks_q)
            self.results_q_s.append(results_q)
        
            for j in range(self.num_workers):
                new_seed = seed * 1000 + i  # Give each actor a uniquely seeded env
                self.actors.append(Actor(tasks_q, results_q, env_id, make_env, new_seed, max_timesteps_per_episode))

        for a in self.actors:
            a.start()

        # we will start by running 20,000 / 1000 = 20 episodes for the first iteration  TODO OLD
        self.average_timesteps_in_episode = 1000

        # number of NN for reward function
        self.num_r = num_r

    def rollout(self, timesteps):
        start_time = time()
        # keep 20,000 timesteps per update  TODO OLD
        num_rollouts = int(timesteps / self.average_timesteps_in_episode)

        for i in range(self.num_policy+1):
            for _ in range(num_rollouts):
                self.tasks_q_s[i].put("do_rollout")
            self.tasks_q_s[i].join()

        paths = []
        paths_scores = np.zeros((num_rollouts))
        score_threshold = 0.5
        time_step_to_start_exploration = 500

        for i in range(num_rollouts):

            if(self._timesteps_elapsed>time_step_to_start_exploration):
                exploration_paths = []

                for j in range(self.num_policy):
                    explooration_path = self.results_q_s[j].get()
                    exploration_paths.append(explooration_path)
                    
                    explooration_path["original_rewards"] = explooration_path["rewards"]
                    explooration_path["rewards"] = self.predictor.predict_reward(explooration_path)

                    paths.append(explooration_path)

                path = self.results_q_s[-1].get()
                ################################
                #  START REWARD MODIFICATIONS  #
                ################################
                path["original_rewards"] = path["rewards"]
                path["rewards"] = self.predictor.predict_reward(path)
                self.predictor.path_callback_explore(path, exploration_paths)
                ################################
                #   END REWARD MODIFICATIONS   #
                ################################


            else:
                path = self.results_q_s[-1].get()
                ################################
                #  START REWARD MODIFICATIONS  #
                ################################
                path["original_rewards"] = path["rewards"]
                path["rewards"] = self.predictor.predict_reward(path)
                self.predictor.path_callback(path)
                ################################
                #   END REWARD MODIFICATIONS   #
                ################################


            # get a segment from the path
            init_seg = sample_segment_from_path(path, int(self.predictor._frames_per_segment))

            paths_scores[i] = self.predictor.predict_segment_quality(init_seg)[1]


            paths.append(path)


        num_good_paths = 4
        if(self._timesteps_elapsed<500): # to use critique 
            # # choose the best paths among the givens
            idx_good_paths = paths_scores.argsort()[::-1][0:num_good_paths]

            paths = [paths[i] for i in idx_good_paths]


        # to find the elapsed time
        for path in paths:
            self._timesteps_elapsed += len(path)



        self.average_timesteps_in_episode = sum([len(path["rewards"]) for path in paths]) / len(paths)

        return paths, time() - start_time

    def set_policy_weights(self, parameters):
        # parameters is a list of all the explorations weights from policy
        # the last one is related to the main policy
        for i in range(self.num_policy+1):
            for j in range(self.num_workers):
                self.tasks_q_s[i].put(parameters[i])
            self.tasks_q_s[i].join()

    def end(self):
        for i in range(self.num_policy+1):
            for j in range(self.num_workers):
                self.tasks_q_s[i].put("kill")
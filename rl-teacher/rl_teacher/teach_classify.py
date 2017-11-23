import os
import os.path as osp
import random
from collections import deque
from time import time, sleep

import numpy as np
import tensorflow as tf
from keras import backend as K
from parallel_trpo.train import train_parallel_trpo
from pposgd_mpi.run_mujoco import train_pposgd_mpi

from rl_teacher.comparison_collectors import SyntheticComparisonCollector, HumanComparisonCollector
from rl_teacher.envs import get_timesteps_per_episode
from rl_teacher.envs import make_with_torque_removed
from rl_teacher.label_schedules import LabelAnnealer, ConstantLabelSchedule
from rl_teacher.nn import FullyConnectedMLP, FullyConnected_classifier
from rl_teacher.segment_sampling import sample_segment_from_path
from rl_teacher.segment_sampling import segments_from_rand_rollout
from rl_teacher.summaries import AgentLogger, make_summary_writer
from rl_teacher.utils import slugify, corrcoef
from rl_teacher.video import SegmentVideoRecorder

CLIP_LENGTH = 1.5

class TraditionalRLRewardPredictor(object):
    """Predictor that always returns the true reward provided by the environment."""

    def __init__(self, summary_writer):
        self.agent_logger = AgentLogger(summary_writer)

    def predict_reward(self, path):
        self.agent_logger.log_episode(path)  # <-- This may cause problems in future versions of Teacher.
        return path["original_rewards"]

    def path_callback(self, path):
        pass

class ComparisonRewardPredictor():
# important
    """Predictor that trains a model to predict how much reward is contained in a trajectory segment"""

    def __init__(self, env, summary_writer, comparison_collector, agent_logger, label_schedule, num_r):
        self.summary_writer = summary_writer
        self.agent_logger = agent_logger
        self.comparison_collector = comparison_collector
        self.label_schedule = label_schedule
        self.num_r = num_r

        # Set up some bookkeeping
        self.recent_segments = deque(maxlen=200)  # Keep a queue of recently seen segments to pull new comparisons from
        self._frames_per_segment = CLIP_LENGTH * env.fps
        self._steps_since_last_training = 0
        self._n_timesteps_per_predictor_training = 1e2  # How often should we train our predictor?
        self._elapsed_predictor_training_iters = 0

        # Build and initialize our predictor model
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        self.sess = tf.InteractiveSession(config=config)
        self.obs_shape = env.observation_space.shape
        self.discrete_action_space = not hasattr(env.action_space, "shape")
        self.act_shape = (env.action_space.n,) if self.discrete_action_space else env.action_space.shape
        self.graph = self._build_model()
        self.sess.run(tf.global_variables_initializer())

    def _predict_rewards(self, obs_segments, act_segments, network):
        """
        :param obs_segments: tensor with shape = (batch_size, segment_length) + obs_shape
        :param act_segments: tensor with shape = (batch_size, segment_length) + act_shape
        :param network: neural net with .run() that maps obs and act tensors into a (scalar) value tensor
        :return: tensor with shape = (batch_size, segment_length)
        """
        batchsize = tf.shape(obs_segments)[0]
        segment_length = tf.shape(obs_segments)[1]

        # Temporarily chop up segments into individual observations and actions
        obs = tf.reshape(obs_segments, (-1,) + self.obs_shape)
        acts = tf.reshape(act_segments, (-1,) + self.act_shape)
        # Run them through our neural network
        rewards = network.run(obs, acts)

        # Group the rewards back into their segments
        return tf.reshape(rewards, (batchsize, segment_length))



    def _build_model(self):
        """
        Our model takes in path segments with states and actions, and generates Q values.
        These Q values serve as predictions of the true reward.
        We can compare two segments and sum the Q values to get a prediction of a label
        of which segment is better. We then learn the weights for our model by comparing
        these labels with an authority (either a human or synthetic labeler).
        """
        # Set up observation placeholders
        self.segment_obs_placeholder = tf.placeholder(
            dtype=tf.float32, shape=(None, None) + self.obs_shape, name="obs_placeholder")
        self.segment_alt_obs_placeholder = tf.placeholder(
            dtype=tf.float32, shape=(None, None) + self.obs_shape, name="alt_obs_placeholder")

        self.segment_act_placeholder = tf.placeholder(
            dtype=tf.float32, shape=(None, None) + self.act_shape, name="act_placeholder")
        self.segment_alt_act_placeholder = tf.placeholder(
            dtype=tf.float32, shape=(None, None) + self.act_shape, name="alt_act_placeholder")


        # A vanilla multi-layer perceptron maps a (state, action) pair to a reward (Q-value)
        # make a list for reward networks
        mlps = []
        self.q_values = []
        self.loss_ops = []
        self.train_ops = []
        self.labels = tf.placeholder(dtype=tf.int32, shape=(None,), name="comparison_labels")
        # loop over the num_r to cluster of NNs
        for i in range(self.num_r):
            # NN for each reward
            mlp = FullyConnectedMLP(self.obs_shape, self.act_shape)
            mlps.append(mlp)
            # q_vlaue and alt_q_value for each reward network
            q_value = self._predict_rewards(self.segment_obs_placeholder, self.segment_act_placeholder, mlp)
            self.q_values.append(q_value)

            alt_q_value = self._predict_rewards(self.segment_alt_obs_placeholder, self.segment_alt_act_placeholder, mlp)

#       mlp = FullyConnectedMLP(self.obs_shape, self.act_shape)

#        self.q_value = self._predict_rewards(self.segment_obs_placeholder, self.segment_act_placeholder, mlp)
#        alt_q_value = self._predict_rewards(self.segment_alt_obs_placeholder, self.segment_alt_act_placeholder, mlp)

            # We use trajectory segments rather than individual (state, action) pairs because
            # video clips of segments are easier for humans to evaluate
            segment_reward_pred_left = tf.reduce_sum(q_value, axis=1)
            segment_reward_pred_right = tf.reduce_sum(alt_q_value, axis=1)
            reward_logits = tf.stack([segment_reward_pred_left, segment_reward_pred_right], axis=1)  # (batch_size, 2)



            # delta = 1e-5
            # clipped_comparison_labels = tf.clip_by_value(self.comparison_labels, delta, 1.0-delta)

            data_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=reward_logits, labels=self.labels)

            loss_op = tf.reduce_mean(data_loss)
            self.loss_ops.append(loss_op)

            global_step = tf.Variable(0, name='global_step', trainable=False)
            train_op = tf.train.AdamOptimizer().minimize(loss_op, global_step=global_step)
            self.train_ops.append(train_op)


        # segment quality classifier
        # placeholder for the concatenated obs and act
        self.segment_placeholder = tf.placeholder(
            dtype=tf.float32, shape=(None,(np.prod(self.obs_shape)+np.prod(self.act_shape))*self._frames_per_segment), name="input_placeholder_classifier")

        # model
        mlp_classifier = FullyConnected_classifier(self.obs_shape, self.act_shape, self._frames_per_segment)

        # labels from human
        self.labels_from_human = tf.placeholder(dtype=tf.int32, shape=(None,), name="softmax_labels")

        # raw output from the classifier
        self.softmax_predicted_labels = mlp_classifier.run(self.segment_placeholder)
        # loss for classifier
        self.loss_softmax_classifier = tf.nn.sparse_softmax_cross_entropy_with_logits(logits= self.softmax_predicted_labels, labels=self.labels_from_human)


        self.train_softmax_classifier = tf.train.AdamOptimizer().minimize(self.loss_softmax_classifier)


        return tf.get_default_graph()

    # combine the state and actions for a segment
    def obs_act_combine(self, obs, act):
        return np.concatenate(( obs.reshape(1,-1) ,act.reshape(1,-1)), axis =1)

    # to predict the qulaity of a given segment
    def predict_segment_quality(self, segment):
        obs_act = self.obs_act_combine(segment['obs'], segment['actions'])
        with self.graph.as_default():
            scores = self.sess.run(tf.nn.softmax(self.softmax_predicted_labels), feed_dict = {self.segment_placeholder: obs_act})

            return scores[0]



    def predict_reward(self, path):
        """Predict the reward for each step in a given path"""
        q_value_total = 0
        with self.graph.as_default():
            for i in range(self.num_r):
                q_value = self.sess.run(self.q_values[i], feed_dict={
                    self.segment_obs_placeholder: np.asarray([path["obs"]]),
                    self.segment_act_placeholder: np.asarray([path["actions"]]),
                    K.learning_phase(): False
                })
                q_value_total += q_value[0]

            return q_value_total/self.num_r

    def predict_segment_individual_reward(self, segment, r_idx):
        """Predict the reward for each step in a given path"""
        with self.graph.as_default():
            q_value = self.sess.run(self.q_values[r_idx], feed_dict={
                self.segment_obs_placeholder: np.asarray([segment["obs"]]),
                self.segment_act_placeholder: np.asarray([segment["actions"]]),
                K.learning_phase(): False
            })
        return np.sum(q_value)
#   def predict_reward(self, path):
#        """Predict the reward for each step in a given path"""
#        with self.graph.as_default():
#            q_value = self.sess.run(self.q_value, feed_dict={
#                self.segment_obs_placeholder: np.asarray([path["obs"]]),
#                self.segment_act_placeholder: np.asarray([path["actions"]]),
#                K.learning_phase(): False
#            })
#        return q_value[0]

    def path_callback(self, path):
        path_length = len(path["obs"])
        self._steps_since_last_training += path_length/2.0

        self.agent_logger.log_episode(path)

        
        # We may be in a new part of the environment, so we take new segments to build comparisons from
        segment = sample_segment_from_path(path, int(self._frames_per_segment))
        if segment:
            self.recent_segments.append(segment)

        score_threshold = 0.8
        # If we need more comparisons, then we build them from our recent segments
        if len(self.comparison_collector) < int(self.label_schedule.n_desired_labels) and len(self.recent_segments)>10:
            # generate segments
            num_sampled_segments = 20
            sampeled_segments = []
            sampeled_segments_score = np.zeros((num_sampled_segments))
            for i in range(num_sampled_segments):
                sampled_segment = random.choice(self.recent_segments)
                sampeled_segments.append(sampled_segment)
                sampeled_segments_score[i] = self.predict_segment_quality(sampled_segment)[1]

            good_segment = sampeled_segments[np.argmax(sampeled_segments_score)]
            bad_segment  = sampeled_segments[np.argmin(sampeled_segments_score)]


            # find the good segment
            # no_seg = True
            # while no_seg:
            #     sample_seg = random.choice(self.recent_segments)
            #     obs_act_seg = self.obs_act_combine(sample_seg['obs'], sample_seg['actions'])
            #     if (predict_segment_quality[1] > 0.5):
            #         good_segment = sample_seg
            #         no_seg = False

            # # find the bad segment
            # no_seg = True
            # while no_seg:
            #     sample_seg = random.choice(self.recent_segments)
            #     obs_act_seg = self.obs_act_combine(sample_seg['obs'], sample_seg['actions'])
            #     if (predict_segment_quality[1] < 0.5):
            #         bad_segment = sample_seg
            #         no_seg = False


            if(np.max(sampeled_segments_score)>score_threshold):
                self.comparison_collector.add_segment_pair_with_label(good_segment, bad_segment, 0)
            else:
                self.comparison_collector.add_segment_pair(good_segment, bad_segment)

            # 
        #     n_cand_pairs = 1
        #     cand_pairs_idx = np.random.randint(len(self.recent_segments), size=(n_cand_pairs, 2))
        #     cand_pairs = []
        #     segment_pairs = []
        #     for i in range(n_cand_pairs):
        #         segment_pair = {}
        #         segment_pair['segment1'] = self.recent_segments[cand_pairs_idx[i,0]]
        #         segment_pair['segment2'] = self.recent_segments[cand_pairs_idx[i,1]]
                
        #         which_seg = np.zeros((self.num_r))
        #         for j in range(self.num_r):
        #             which_seg[j] = self.predict_segment_individual_reward(segment_pair['segment1'], j)> self.predict_segment_individual_reward(segment_pair['segment2'], j)
                
        #         segment_pair['std'] = np.std(which_seg)
        #         segment_pairs.append(segment_pair)

        #     max_std_idx = np.argmax(np.array([pair_std['std'] for pair_std in segment_pairs]))
        #     chosen_pair = segment_pairs[max_std_idx]
        #     self.comparison_collector.add_segment_pair(
        #         chosen_pair['segment1'],chosen_pair['segment2'])
        # Train our predictor every X steps
        if self._steps_since_last_training >= int(self._n_timesteps_per_predictor_training):
            # sleep(5)
            self.train_predictor()
            self._steps_since_last_training -= self._steps_since_last_training

    def path_callback_explore(self, path, other_paths):
        path_length = len(path["obs"])
        self._steps_since_last_training += path_length/2.0

        self.agent_logger.log_episode(path)

        
        # We may be in a new part of the environment, so we take new segments to build comparisons from
        segment = sample_segment_from_path(path, int(self._frames_per_segment))
        if segment:
            self.recent_segments.append(segment)

        for i in range(len(other_paths)):
            other_segment = sample_segment_from_path(other_paths[i], int(self._frames_per_segment))
            if other_segment and segment:
                self.comparison_collector.add_segment_pair(segment, other_segment)
        # Train our predictor every X steps
        if self._steps_since_last_training >= int(self._n_timesteps_per_predictor_training):
            # sleep(2)
            self.train_predictor()
            self._steps_since_last_training -= self._steps_since_last_training

    def train_predictor(self):
        self.comparison_collector.label_unlabeled_comparisons()

        # train segment classifier
        # labeled sgements
        # minibatch_size = min(64, len(self.comparison_collector.labeled_soft_decisive_comparisons))
        # labeled_comparisons = random.sample(self.comparison_collector.labeled_soft_decisive_comparisons, minibatch_size)
        minibatch_size = min(64, len(self.comparison_collector._comparisons_labeled_soft))
        labeled_comparisons = random.sample(self.comparison_collector._comparisons_labeled_soft, minibatch_size)

        segments_to_classify_left   = np.concatenate( [self.obs_act_combine(labeled_comparisons[i]['left']['obs'] ,  labeled_comparisons[i]['left']['actions']) for i in range(len(labeled_comparisons))])
        segments_to_classify_right  = np.concatenate( [self.obs_act_combine(labeled_comparisons[i]['right']['obs'] ,labeled_comparisons[i]['right']['actions']) for i in range(len(labeled_comparisons))])

        # segments_to_classify_left  = np.concatenate( [np.concatenate(( labeled_comparisons[i]['left']['obs'].reshape(1,-1) ,labeled_comparisons[i]['left']['actions'].reshape(1,-1)), axis =1) for i in range(len(labeled_comparisons))])
        # segments_to_classify_right = np.concatenate( [np.concatenate(( labeled_comparisons[i]['right']['obs'].reshape(1,-1) ,labeled_comparisons[i]['right']['actions'].reshape(1,-1)), axis =1) for i in range(len(labeled_comparisons))])
        segments_to_classify       = np.concatenate( (segments_to_classify_left , segments_to_classify_right), axis = 0)



        segments_labels_to_classify_left  = np.concatenate([[1 if segment['label']==0 or 2 else 0] for segment in labeled_comparisons])
        segments_labels_to_classify_right = np.concatenate([[1 if segment['label']==1 or 2 else 0] for segment in labeled_comparisons])
        segments_labels_to_classify       = np.concatenate((segments_labels_to_classify_left, segments_labels_to_classify_right), axis=0)

        with self.graph.as_default():
            self.sess.run(self.train_softmax_classifier, feed_dict = {
                self.segment_placeholder : segments_to_classify,
                self.labels_from_human : segments_labels_to_classify
                })


        # train the reward function
        
        for i in range(self.num_r):
            # minibatch_size = min(64, len(self.comparison_collector.labeled_decisive_comparisons))
            # labeled_comparisons = random.sample(self.comparison_collector.labeled_decisive_comparisons, minibatch_size)
            minibatch_size = min(64, len(self.comparison_collector._comparisons_labeled))
            labeled_comparisons = random.sample(self.comparison_collector._comparisons_labeled, minibatch_size)
            left_obs = np.asarray([comp['left']['obs'] for comp in labeled_comparisons])
            left_acts = np.asarray([comp['left']['actions'] for comp in labeled_comparisons])
            right_obs = np.asarray([comp['right']['obs'] for comp in labeled_comparisons])
            right_acts = np.asarray([comp['right']['actions'] for comp in labeled_comparisons])
            labels = np.asarray([comp['label'] for comp in labeled_comparisons])


            with self.graph.as_default():
                    _, loss = self.sess.run([self.train_ops[i], self.loss_ops[i]], feed_dict={
                        self.segment_obs_placeholder: left_obs,
                        self.segment_act_placeholder: left_acts,
                        self.segment_alt_obs_placeholder: right_obs,
                        self.segment_alt_act_placeholder: right_acts,
                        self.labels: labels,
                        K.learning_phase(): True
                        })
            self._elapsed_predictor_training_iters += 1
            self._write_training_summaries(loss)

    def _write_training_summaries(self, loss):
        self.agent_logger.log_simple("predictor/loss", loss)

        # Calculate correlation between true and predicted reward by running validation on recent episodes
        recent_paths = self.agent_logger.get_recent_paths_with_padding()
        if len(recent_paths) > 1 and self.agent_logger.summary_step % 10 == 0:  # Run validation every 10 iters
            validation_obs = np.asarray([path["obs"] for path in recent_paths])
            validation_acts = np.asarray([path["actions"] for path in recent_paths])
            q_value = self.sess.run(self.q_values[0], feed_dict={
                self.segment_obs_placeholder: validation_obs,
                self.segment_act_placeholder: validation_acts,
                K.learning_phase(): False
            })
            ep_reward_pred = np.sum(q_value, axis=1)
            reward_true = np.asarray([path['original_rewards'] for path in recent_paths])
            ep_reward_true = np.sum(reward_true, axis=1)
            self.agent_logger.log_simple("predictor/correlations", corrcoef(ep_reward_true, ep_reward_pred))

        self.agent_logger.log_simple("predictor/num_training_iters", self._elapsed_predictor_training_iters)
        self.agent_logger.log_simple("labels/desired_labels", self.label_schedule.n_desired_labels)
        self.agent_logger.log_simple("labels/total_comparisons", len(self.comparison_collector))
        self.agent_logger.log_simple(
            "labels/labeled_comparisons", len(self.comparison_collector.labeled_decisive_comparisons))

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env_id', required=True)
    parser.add_argument('-p', '--predictor', required=True)
    parser.add_argument('-n', '--name', required=True)
    parser.add_argument('-s', '--seed', default=1, type=int)
    parser.add_argument('-w', '--workers', default=2, type=int)
    parser.add_argument('-l', '--n_labels', default=None, type=int)
    parser.add_argument('-L', '--pretrain_labels', default=None, type=int)
    parser.add_argument('-t', '--num_timesteps', default=5e6, type=int)
    parser.add_argument('-a', '--agent', default="parallel_trpo", type=str)
    parser.add_argument('-i', '--pretrain_iters', default=10000, type=int)
    parser.add_argument('-V', '--no_videos', action="store_true")
    args = parser.parse_args()

    num_r = 1
    print("Setting things up...")

    env_id = args.env_id
    run_name = "%s/%s-%s" % (env_id, args.name, int(time()))
    summary_writer = make_summary_writer(run_name)

    env = make_with_torque_removed(env_id)

    num_timesteps = int(args.num_timesteps)
    experiment_name = slugify(args.name)

    if args.predictor == "rl":
        predictor = TraditionalRLRewardPredictor(summary_writer)
    else:
        agent_logger = AgentLogger(summary_writer)

        pretrain_labels = args.pretrain_labels if args.pretrain_labels else args.n_labels // 4

        if args.n_labels:
            label_schedule = LabelAnnealer(
                agent_logger,
                final_timesteps=num_timesteps,
                final_labels=args.n_labels,
                pretrain_labels=pretrain_labels)
        else:
            print("No label limit given. We will request one label every few seconds.")
            label_schedule = ConstantLabelSchedule(pretrain_labels=pretrain_labels)

        if args.predictor == "synth":
            comparison_collector = SyntheticComparisonCollector()

        elif args.predictor == "human":
            bucket = os.environ.get('RL_TEACHER_GCS_BUCKET')
            assert bucket and bucket.startswith("gs://"), "env variable RL_TEACHER_GCS_BUCKET must start with gs://"
            comparison_collector = HumanComparisonCollector(env_id, experiment_name=experiment_name)
        else:
            raise ValueError("Bad value for --predictor: %s" % args.predictor)

        predictor = ComparisonRewardPredictor(
            env,
            summary_writer,
            comparison_collector=comparison_collector,
            agent_logger=agent_logger,
            label_schedule=label_schedule,
            num_r = num_r
        )

        print("Starting random rollouts to generate pretraining segments. No learning will take place...")
        pretrain_segments = segments_from_rand_rollout(
            env_id, make_with_torque_removed, n_desired_segments=pretrain_labels * 2,
            clip_length_in_seconds=CLIP_LENGTH, workers=args.workers)
        for i in range(pretrain_labels):  # Turn our random segments into comparisons
            comparison_collector.add_segment_pair(pretrain_segments[i], pretrain_segments[i + pretrain_labels])

        # Sleep until the human has labeled most of the pretraining comparisons
        while len(comparison_collector.labeled_comparisons) < int(pretrain_labels * 0.75):
            comparison_collector.label_unlabeled_comparisons()
            if args.predictor == "synth":
                print("%s synthetic labels generated... " % (len(comparison_collector.labeled_comparisons)))
            elif args.predictor == "human":
                print("%s/%s comparisons labeled. Please add labels w/ the human-feedback-api. Sleeping... " % (
                    len(comparison_collector.labeled_comparisons), pretrain_labels))
                sleep(5)

        # Start the actual training
        for i in range(args.pretrain_iters):
            predictor.train_predictor()  # Train on pretraining labels
            if i % 100 == 0:
                print("%s/%s predictor pretraining iters... " % (i, args.pretrain_iters))

    # Wrap the predictor to capture videos every so often:
    if not args.no_videos:
        predictor = SegmentVideoRecorder(predictor, env, save_dir=osp.join('/tmp/rl_teacher_vids', run_name))

    # We use a vanilla agent from openai/baselines that contains a single change that blinds it to the true reward
    # The single changed section is in `rl_teacher/agent/trpo/core.py`
    print("Starting joint training of predictor and agent")
    if args.agent == "parallel_trpo":
        train_parallel_trpo(
            env_id=env_id,
            make_env=make_with_torque_removed,
            predictor=predictor,
            num_r = num_r,
            summary_writer=summary_writer,
            workers=args.workers,
            runtime=(num_timesteps),
            max_timesteps_per_episode=get_timesteps_per_episode(env),
            timesteps_per_batch=8000,
            max_kl=0.001,
            seed=args.seed,
            num_policy = 3,
            exploration = True,
        )
    elif args.agent == "pposgd_mpi":
        def make_env():
            return make_with_torque_removed(env_id)

        train_pposgd_mpi(make_env, num_timesteps=num_timesteps, seed=args.seed, predictor=predictor)
    else:
        raise ValueError("%s is not a valid choice for args.agent" % args.agent)

if __name__ == '__main__':
    main()

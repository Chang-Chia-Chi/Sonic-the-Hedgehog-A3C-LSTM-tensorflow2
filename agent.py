import os
import cv2
import gym
import random
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from parser import parser
from utils import *
from Actions import ACTIONS
from multiprocessing import cpu_count
from threading import Thread, Lock
from model import ActorCritic
from env_wrap import make_sonic_env

args = parser.parse_args()
COORD = tf.train.Coordinator()
class Agent:
    def __init__(self, n_units, pretrained=False, weights_path=None, conv_fn="impala_cnn"):
        # self.state_dim = env.observation_space.shape
        self.action_dim = len(ACTIONS)
        self.n_units = n_units
        self.conv_fn = conv_fn

        self.global_worker = ActorCritic(self.action_dim, self.n_units, pretrained=pretrained, weights_path=weights_path, conv_fn=self.conv_fn)
        self.num_workers = args.workers
        self.lock = Lock()

    def fit(self, max_episodes=args.max_episodes):
        envs = []
        workers = []

        num_sample = self.num_workers - args.num_non_sample
        for i in range(self.num_workers):
            env = make_sonic_env(game=args.game, state=args.state)
            envs.append(env)
            if i < num_sample:
                workers.append(Worker(i, env, self.n_units, self.global_worker, self.lock, max_episodes, sample=True, conv_fn=self.conv_fn))
            else:
                workers.append(Worker(i, env, self.n_units, self.global_worker, self.lock, max_episodes, sample=False, conv_fn=self.conv_fn))

        for worker in workers:
            worker.start()
        
        COORD.join(workers)

class Worker(Thread):
    def __init__(self, seed, env, n_units, global_worker, lock, max_episodes, sample=True, conv_fn="impala_cnn"):
        Thread.__init__(self)
        self.seed = seed
        self.env = env
        self.action_dim = len(ACTIONS)
        self.lock = lock
        self.conv_fn = conv_fn
        self.save = True if seed == 0 else False
        self.sample = sample

        self.global_worker = global_worker
        self.max_episodes = max_episodes

        self.n_units = n_units
        self.ACNet = ActorCritic(self.action_dim, self.n_units, conv_fn=self.conv_fn)
        self.model = self.ACNet.model

        # initialization of weights to be the same as global network
        self.pull_param()
    
    def pull_param(self):
        """
        pull parameters from global network

        Reference:
        https://github.com/iverxin/rl_impl
        """
        for worker_para, global_para in zip(self.model.trainable_variables, self.global_worker.model.trainable_variables):
            worker_para.assign(global_para)

    def fit(self):
        tf.random.set_seed(self.seed + args.seed)
        print('Process {} start training'.format(self.seed))
        save_path = os.getcwd()+'/weights/training/sonic_weight'
        episode_count = 0
        while not COORD.should_stop():
            if episode_count > self.max_episodes:
                print('Process {} done trainning'.format(self.seed))
                break
            if self.save and (episode_count % args.save_interval == 0) and episode_count > 0:
                with self.lock:
                    self.model.save_weights(save_path)
                    print('Saving model weights at episode {}'.format(episode_count))
            new_ep = True
            state = self.env.reset()
            state = tf.cast(np.expand_dims(state, axis=0), dtype=tf.float32)
            
            ep_reward = 0
            done = False
            while not done:
                if new_ep:
                    ht = tf.zeros((1, self.n_units))
                    ct = tf.zeros((1, self.n_units))
                    new_ep = False
                
                rewards = []
                values = []
                log_probs = []
                entropies = []
                with tf.GradientTape() as tape:
                    for step in range(args.time_step):
                        act_score, value, (ht, ct) = self.model((state, (ht, ct)))
                        # turn action score to probability distribution
                        dist = tfp.distributions.Categorical(logits=act_score)
                        
                        if self.sample:
                            action = dist.sample()
                        else:
                            action = tf.math.argmax(tf.math.softmax(act_score[0]))
                            action = tf.expand_dims(action, axis=-1)

                        log_prob = dist.log_prob(action)
                        entropy = dist.entropy()

                        state, reward, done, info = self.env.step(action[0])
                        state = tf.cast(np.expand_dims(state, axis=0), dtype=tf.float32)

                        ep_reward += reward
                        rewards.append(reward)
                        values.append(value)
                        log_probs.append(log_prob)
                        entropies.append(entropy)
                        
                        if done:
                            break
                    
                    if not done:
                        _, value, (_, _) = self.model((state, (ht, ct)))
                        last_value = value
                    else:
                        last_value = tf.zeros((1, 1))
                    
                    q_values = [last_value]
                    for i in reversed(range(len(rewards))):
                        q_values.insert(0, args.gamma*q_values[0]+rewards[i])

                    policy_loss = 0.0
                    critic_loss = 0.0
                    entropy_loss = 0.0
                    for i in reversed(range(len(rewards))):
                        adv = tf.stop_gradient(q_values[i]) - values[i]

                        critic_loss += tf.reduce_mean(0.5*tf.square(adv))
                        policy_loss += -tf.reduce_mean(tf.stop_gradient(adv)*log_probs[i])
                        entropy_loss += tf.reduce_mean(entropies[i])

                    total_loss = policy_loss + args.critic_loss_coef*critic_loss - args.entropy_coef*entropy_loss
                
                with self.lock:
                    grads = tape.gradient(total_loss, self.model.trainable_variables)
                    grads, _ = tf.clip_by_global_norm(grads, args.max_grad_norm)
                    self.global_worker.opt.apply_gradients(zip(grads, self.global_worker.model.trainable_variables))
                    self.pull_param()

                print('Process {}/X: {}/Episode: {}/EP_Reward: {:.2f}/Loss: {:.3f}'.format(self.seed, info['x'], episode_count, ep_reward, total_loss))

            text_info = '\n##Process {}: Episode {} get rewards {:2f} with dist {}##\n'.format(self.seed, episode_count, ep_reward, info['x'])
            ep_reward = 0
            print(text_info)
            with self.lock:
                episode_count += 1

    def run(self):
        self.fit()
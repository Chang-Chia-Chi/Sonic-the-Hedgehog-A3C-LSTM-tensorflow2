import os
import gym
import retro
import retrowrapper
import numpy as np
import tensorflow as tf
from Actions import ACTIONS
from model import ActorCritic
from env_wrap import make_sonic_env

weights_path = os.getcwd()+'/weights/GreenHillZone_Act1/sonic_weight'
env = make_sonic_env(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1', test=True, record=True)
model = ActorCritic(len(ACTIONS), 512).model

model.load_weights(weights_path)
state = env.reset()
state = tf.cast(np.expand_dims(state, axis=0), dtype=tf.float32)
            
ep_reward = 0
done = False
new_ep = True
while not done:
    env.render()
    if new_ep:
        ht = tf.zeros((1, 512))
        ct = tf.zeros((1, 512))
        new_ep = False
    
    act_score, value, (ht, ct) = model((state, (ht, ct)))
    action = tf.math.argmax(tf.math.softmax(act_score[0]))

    state, reward, done, info = env.step(action)
    state = tf.cast(np.expand_dims(state, axis=0), dtype=tf.float32)
    print(info['x'])
    ep_reward += reward

env.close()

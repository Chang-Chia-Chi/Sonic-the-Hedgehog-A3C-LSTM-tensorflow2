import cv2
import gym
import retro
import retrowrapper
import numpy as np
from collections import deque

"""
Enviroment wrappers for Sonic training.

Reference:
https://github.com/openai/baselines/
https://github.com/sadeqa/Super-Mario-Bros-RL/
https://github.com/MaxStrange/retrowrapper
https://note.com/npaka/n/n9a0693306035
https://retro.readthedocs.io/en/latest/python.html
"""

class StochasticFrameSkip(gym.Wrapper):
    """
    Use same actions for n frames, with probability = stickprob 
    that first frame will use action from previous step

    The purpose is to reduce times of decision making so that 
    decrease time required per episode

    It's very useful for games with repeated actions
    """
    def __init__(self, env, n, stickprob):
        gym.Wrapper.__init__(self, env)
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()
        self.supports_want_render = hasattr(env, "supports_want_render")

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac):
        done = False
        totrew = 0
        for i in range(self.n):
            # First step after reset, use action
            if self.curac is None:
                self.curac = ac
            # First substep, delay with probability=stickprob
            elif i==0:
                if self.rng.rand() > self.stickprob:
                    self.curac = ac
            # Second substep, new action definitely kicks in
            elif i==1:
                self.curac = ac
            if self.supports_want_render and i<self.n-1:
                ob, rew, done, info = self.env.step(self.curac, want_render=False)
            else:
                ob, rew, done, info = self.env.step(self.curac)
            totrew += rew
            if done: break
        return ob, totrew, done, info

    def seed(self, s):
        self.rng.seed(s)

class TimeLimit(gym.Wrapper):
    """
    Restrict steps per episode
    """
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

class ProcessFrame(gym.ObservationWrapper):
    """
    pre-process frame, resizing and convert to specified color
    """
    def __init__(self, env, width=84, height=84, grayscale=True):
        super(ProcessFrame, self).__init__(env)
        self.width = width
        self.height = height
        self.grayscale = grayscale

        if self.grayscale == True:
            self.channel = 1
        else:
            self.channel = 3

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.height, self.width, self.channel), dtype=np.uint8)

    def observation(self, obs):
        frame = obs
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = np.expand_dims(frame, axis=-1)
        frame = cv2.resize(frame, (self.width, self.height))
        obs = frame
        return obs/255.

class FrameStack(gym.Wrapper):
    """
    Stack k last frames to make agent running more efficiently
    """
    def __init__(self, env, k):
        super(FrameStack, self).__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(shape[0], shape[1], shape[2]*k), dtype=env.observation_space.dtype)
        self.height = shape[0]
        self.width = shape[0]
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        total_reward = reward
        self.frames.append(obs)
        for i in range(self.k-1):
            if not done:
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
                self.frames.append(obs)
            else:
                self.frames.append(obs)
        
        frames = np.stack(self.frames, axis=0)
        frames = np.reshape(frames, (self.height, self.width, self.k))
        return frames, total_reward, done, info
    
    def reset(self):
        self.frames.clear()
        obs = self.env.reset()
        for i in range(self.k):
            self.frames.append(obs)

        frames = np.stack(self.frames, axis=0)
        frames = np.reshape(frames, (self.height, self.width, self.k))
        return frames

class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):
        return self._actions[a].copy()

class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.
    This is incredibly important and effects performance
    drastically.
    """
    def __init__(self, env, scale=0.01):
        super(RewardScaler, self).__init__(env)
        self.scale = scale

    def reward(self, reward):
        return reward * self.scale

class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.

    (restrict punishments if AI goes back)
    """
    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0

    def reset(self, **kwargs):
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, rew, done, info

def sonic_wrapper(game=None, 
                  state=None, 
                  height=84, 
                  width=84, 
                  grayscale=True, 
                  stack=True,
                  test=False,
                  record=False,
                  k=4, 
                  scale_rew=True, 
                  scale=0.01, 
                  allowbacktrace=True):

    use_restricted_actions = retro.Actions.FILTERED
    if record:
        env = retro.make(game=game, state=state, scenario='contest', use_restricted_actions=use_restricted_actions, record='.')
    else:
        env = retro.make(game=game, state=state, scenario='contest', use_restricted_actions=use_restricted_actions)
    env = SonicDiscretizer(env)

    if test:
        stickprob = 0
    else:
        stickprob = 0.25
    env = StochasticFrameSkip(env, n=4, stickprob=stickprob)
    env = ProcessFrame(env, width=width, height=height, grayscale=grayscale)
    env = TimeLimit(env, max_episode_steps=4500)
    if stack:
        env = FrameStack(env, k)
    if scale_rew:
        env = RewardScaler(env, scale)
    if allowbacktrace:
        env = AllowBacktracking(env)
    return env

def make_sonic_env(game=None, state=None, test=False, record=False):
    retrowrapper.set_retro_make(sonic_wrapper)
    env = retrowrapper.RetroWrapper(game=game, state=state, test=test, record=record)
    return env
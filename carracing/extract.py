'''
saves ~ 200 episodes generated from a random policy
'''

import numpy as np
import random
import os
import gym

from env import make_env
from model import make_model

MAX_FRAMES = 1000 # max length of carracing
MAX_TRIALS = 200 # just use this to extract one trial. 

render_mode = False # for debugging.

DIR_NAME = 'record'
if not os.path.exists(DIR_NAME):
    os.makedirs(DIR_NAME)

controller = make_model()

total_frames = 0
env = make_env(render_mode=render_mode, full_episode=False)
for trial in range(MAX_TRIALS): # 200 trials per worker
  try:
    random_generated_int = random.randint(0, 2**31-1)
    filename = DIR_NAME+"/"+str(random_generated_int)+".npz"
    recording_obs = []
    recording_action = []
    recording_reward = []
    recording_done = []

    np.random.seed(random_generated_int)
    env.seed(random_generated_int)

    # random policy
    #controller.init_random_model_params(stdev=np.random.rand()*0.01)
    # strong policy
    controller.load_model('log/carracing.cma.16.64.json')

    obs = env.reset() # pixels

    for frame in range(MAX_FRAMES):
      if render_mode:
        env.render("human")
      else:
        env.render("rgb_array")

      recording_obs.append(obs)
      action = controller.get_action(obs)

      recording_action.append(action)
      obs, reward, done, info = env.step(action)
      recording_reward.append(reward)
      recording_done.append(done)

      if done:
        break

    total_frames += (frame+1)
    print("dead at", frame+1, "total recorded frames for this worker", total_frames)
    recording_obs = np.array(recording_obs, dtype=np.uint8)
    recording_action = np.array(recording_action, dtype=np.float16)
    recording_reward = np.array(recording_reward, dtype=np.float16)
    recording_done = np.array(recording_done, dtype=np.bool)
    np.savez_compressed(filename, obs=recording_obs, action=recording_action, reward=recording_reward, recording_done)
  except gym.error.Error:
    print("stupid gym error, life goes on")
    env.close()
    env = make_env(render_mode=render_mode)
    continue
env.close()

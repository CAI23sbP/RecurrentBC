# from package.testbed.gru.phasic_gru_ppo import Phase1GruPPO , Phase2GruPPO
from stable_baselines3.ppo import PPO
import numpy as np
from package.testbed.gru.gru_bc import GruBC 
from imitation.util.util import make_vec_env

from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from package.testbed.bc.custom_bc import CustomBC
from package.testbed.gru.gru_ppo import GruPPO
from gymnasium.experimental.wrappers.rendering import RecordVideoV0
from gymnasium.experimental.wrappers.common import RecordEpisodeStatisticsV0
import gymnasium as gym
import torch.nn as nn 
import torch 
import random
random.seed(0)

TEST_EPI = 20
MAX_ITER = 500_000
N_STEP = 256
ENV_NAME = "BipedalWalker-v3" # "BipedalWalker-v3" , 
BC_EPISODE = 250

"""
make env
"""
rng = np.random.default_rng(0)
env = make_vec_env(ENV_NAME, 
                   rng= rng, 
                   n_envs = 40, 
                   post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
                   )

"""
make expert
"""
policy_kwargs = dict(
        net_arch = dict(pi = [32,32], vf = [32,32]),
        activation_fn = nn.ReLU,
        )

expert = PPO(policy='MlpPolicy',
            env = env,
            batch_size = 64,
            n_steps=N_STEP,
            policy_kwargs = policy_kwargs,
            n_epochs = 7,
            learning_rate= 0.0004,
            tensorboard_log= '/home/cai/Desktop/PILRnav/tensor_log/expert',
            verbose= 1
            )

expert.learn(MAX_ITER)
expert.save('/home/cai/Desktop/PILRnav/expert')

rollouts = rollout.rollout(
    expert,
    env,
    rollout.make_sample_until(min_timesteps=None, min_episodes = BC_EPISODE),
    rng=rng,
)
transitions = rollout.flatten_trajectories(rollouts)

"""
make PPO learner 
"""
learner = PPO(policy='MlpPolicy',
            env = env,
            batch_size = 64,
            n_steps=N_STEP,
            policy_kwargs = policy_kwargs,
            n_epochs = 7,
            learning_rate= 0.0004,
            tensorboard_log= '/home/cai/Desktop/PILRnav/tensor_log/expert',
            verbose= 1
            )
"""
traininig bc with PPO learner 
"""
origin_bc = CustomBC(observation_space=env.observation_space,
      action_space=env.action_space,
      policy=learner.policy,
    demonstrations = transitions,
    rng= rng,
    device='cuda',
    tensorboard_log = '/home/cai/Desktop/PILRnav/tensor_log/bc')

origin_bc.train(n_epochs = 10)

origin_bc.policy.save('/home/cai/Desktop/PILRnav/bc')

"""
make GRU PPO learner 
"""
policy_kwargs = dict(
        share_features_extractor = True,
        net_arch = dict(pi = [32,32], vf = [32,32]),
        gru_hidden_size = 32,
        activation_fn = nn.ReLU,
        )

learner_gru = GruPPO(policy='MlpGruPolicy',
            env = env,
            batch_size = 64,
            n_steps=N_STEP,
            policy_kwargs = policy_kwargs,
            n_epochs = 7,
            learning_rate= 0.0004,
            tensorboard_log= '/home/cai/Desktop/PILRnav/tensor_log/expert',
            verbose= 1
            )
"""
training GRU bc with GRU PPO 
"""
recurrent_bc_ = GruBC(venv=env, 
                   policy=learner_gru.policy,
                   demonstrations = transitions,
                   rng = rng,
                   device='cuda',
                   tensorboard_log = '/home/cai/Desktop/PILRnav/tensor_log/gru_bc')

recurrent_bc_.train(n_epochs=5, n_batches = 32)
recurrent_bc_.policy.save('/home/cai/Desktop/PILRnav/gru_bc')
del env, rollouts, transitions



epi_rewards = []
epi_lenghts = []
keys = ['expert', 'bc', 'bc_gru']
for index, agent in enumerate([expert, origin_bc ,recurrent_bc_]):
    main_env = gym.make(ENV_NAME, max_episode_steps= 400, render_mode = 'rgb_array')
    env_record = RecordVideoV0(env = main_env, 
                            video_folder = '/home/cai/Desktop/PILRnav/record',
                            video_length = 0,
                            name_prefix = keys[index],
                            episode_trigger = lambda x: x % 4 == 0
                            )
    env_record = RecordEpisodeStatisticsV0(env = env_record, 
                                           )

    for epi in range(TEST_EPI):
        observation, _ = env_record.reset()
        is_done = False
        state = None
        episode_start = np.ones((1,),dtype=bool)
        while not is_done:
            action, state = agent.policy.predict(observation=observation,
                                 state = state,
                                 episode_start= episode_start,
                                 deterministic=True
                                 )
            obs, rewards, dones, time_out, info = env_record.step(action)
            if time_out or dones:
                is_done = True 
                episode_starts =  np.ones((1,),dtype=bool)
            else:
                episode_starts =  np.zeros((1,),dtype=bool)
                is_done = False
    epi_lenghts.append([np.mean(env_record.episode_length_buffer), np.var(env_record.episode_length_buffer), np.std(env_record.episode_length_buffer)])
    epi_rewards.append([np.mean(env_record.episode_reward_buffer), np.var(env_record.episode_length_buffer), np.std(env_record.episode_length_buffer)])
    del env_record
    
import pandas as pd 
epi_infos = {ke:[epi_re,epi_le] for ke, epi_re, epi_le in zip(keys,epi_rewards, epi_lenghts)}
data_frame = pd.DataFrame(epi_infos)
print(data_frame)







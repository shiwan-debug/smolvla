import os
# Remove broken SSL_CERT_FILE so certifi finds its own cacert.pem
os.environ.pop('SSL_CERT_FILE', None)
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

import gym_pusht
import gymnasium as gym
import torch
import numpy as np
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

device = 'cuda'
print('Loading diffusion_pusht policy...')
policy = DiffusionPolicy.from_pretrained('lerobot/diffusion_pusht')
policy = policy.to(device)
policy.eval()
print('Policy loaded and on GPU!')

env = gym.make('gym_pusht/PushT-v0', obs_type='pixels_agent_pos', max_episode_steps=300)
policy.reset()
obs, info = env.reset(seed=42)
print('Environment ready. Running inference...')

for i in range(50):
    state = torch.from_numpy(obs['agent_pos']).float().unsqueeze(0).to(device)
    image = torch.from_numpy(obs['pixels']).float().permute(2,0,1).unsqueeze(0).to(device) / 255.0
    obs_dict = {'observation.state': state, 'observation.image': image}
    with torch.inference_mode():
        action = policy.select_action(obs_dict)
    numpy_action = action.squeeze(0).to('cpu').numpy()
    obs, reward, terminated, truncated, info = env.step(numpy_action)
    if i % 10 == 0:
        print(f'Step {i}: reward={reward:.3f}')
    if terminated or truncated:
        print(f'Episode done at step {i}! terminated={terminated}, truncated={truncated}')
        break
print('Inference complete!')

import math

import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


POSITION_MAX = 0.6
POSITION_MIN = -1.2
VELOCITY_MIN = -0.07
VELOCITY_MAX = 0.07

# Step 1: Setup Environment


env = gym.make('MountainCarContinuous-v0')
state = env.reset()[0]

# Step 2: Initialize Parameters
episodes = 10000
alpha = 0.1
epsilon = 0.3
learning_rate_actor = 0.001
learning_rate_critic = 0.01
gamma = 0.95  # Discount factor
num_episodes = 1000

# Tile coding parameters
num_tiles = 10
num_tilings = 10
tiling_offset = POSITION_MAX-POSITION_MIN/num_tiles

# State space dimensions
state_low = POSITION_MIN  # what?
state_high = POSITION_MAX  # what?
state_dims = env.observation_space.shape[0]

# Initialize weights for actor and critic
actor_weights = np.random.rand(num_tiles*num_tilings)   # 1 was env.action_space.n
critic_weights = np.zeros(num_tiles*num_tilings)


# Step 3: Define Tile-Coding
def tile_code(state):
    position = state[0]
    velocity = state[1]
    # if position < POSITION_MIN:
    #     position = POSITION_MIN
    # if position > POSITION_MAX:
    #     position = POSITION_MAX
    # if velocity < VELOCITY_MIN:
    #     velocity = VELOCITY_MIN
    # if velocity > VELOCITY_MAX:
    #     velocity = VELOCITY_MAX
    tile_width = (POSITION_MAX - POSITION_MIN) / num_tiles
    tile_height = (VELOCITY_MAX - VELOCITY_MIN) / num_tiles
    tiles = np.zeros(num_tilings*num_tiles, dtype=int)
    x = int((position - POSITION_MIN) / tile_width)
    v = int((velocity - VELOCITY_MIN) / tile_height)
    tile = 10*x + v
    tiles[10*x + v] = 1
    return tile


# Step 4: Actor Network
def select_action(state, actor_weights):
    tile = tile_code(state)
    # normal distribution around actor_weights[tile] and between -1 and 1

    distribution_mean = np.cos(actor_weights[tile])
    action = np.random.normal(distribution_mean, 0.2)
    if action > 1:
        action = 1
    if action < -1:
        action = -1
    return [action]


# Step 5: Critic Network
def compute_value(state):
    tile_indices = tile_code(state)
    value = np.sum(critic_weights[tile_indices])
    return value


# Step 6: Policy Gradient
def update_actor_critic(state, action, reward, next_state, done, iteration_num):
    # Compute TD-error and update critic
    td_target = reward + gamma * compute_value(next_state) * (1 - done)
    td_error = td_target - compute_value(state)
    decay_factor = gamma**iteration_num
    critic_gradient = learning_rate_critic * td_error * decay_factor  # add some decay factor
    tile_index = tile_code(state)
    critic_weights[tile_index] += critic_gradient

    # Compute advantage and update actor
    advantages = td_error
    actor_gradient = learning_rate_actor * advantages
    actor_weights[tile_index] += actor_gradient


# Step 7: Training Loop
for episode in tqdm(range(num_episodes)):
    state = env.reset()[0]
    episode_reward = 0

    for t in range(env.spec.max_episode_steps):
        action = select_action(state, actor_weights)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        update_actor_critic(state, action, reward, next_state, done, t)
        state = next_state
        if done:
            break

    # Print episode stats
    print(f"Episode: {episode}, Total Reward: {episode_reward}")

# Step 8: Evaluation
# Evaluate the trained agent's performance





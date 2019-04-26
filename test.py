from unityagents import UnityEnvironment
import numpy as np
import torch
import numpy as np

from replaybuffer import ReplayBuffer
from maddpg_agent import MADDPGAgent

unity_visible = True
if unity_visible:
    tennis_filename = 'Tennis_Linux/Tennis.x86_64'
else:
    tennis_filename = 'Tennis_Linux_NoVis/Tennis.x86_64'

memory_params = {
    'buffer_size': int(1e6),        # replay buffer size
    'batch_size': 128,              # minibatch size
    'seed': 0,                      # Seed to generate random numbers
}

params = {
    'gamma': 0.99,                      # discount factor
    'tau': 0.15,                        # for soft update of target parameters
    'update_every': 1,                  # update parameters per this number
    'lr_actor': 7e-4,                   # learning rate of the Actor
    'lr_critic': 3e-3,                  # learning rate of the Critic
    'seed': 0,                          # Seed to generate random numbers
    'actor_units': [512, 256],          # Number of nodes in hidden layers of the Actor
    'critic_units': [512, 256],         # Number of nodes in hidden layers of the Critic
    'weight_decay': 0,                  # L2 weight decay
    'noise_theta': 0.15,                # Theta of Ornstein-Uhlenbeck process
    'noise_sigma': 0.01,                # Sigma of Ornstein-Uhlenbeck process
}

# Filenames of weights of Actor/Critic
actor_filename = 'model_maddpg_actor'
critic_filename = 'model_maddpg_critic'


def test(agents, max_t=5000):
    env_info = env.reset(train_mode=False)[brain_name] # reset the environment
    states = env_info.vector_observations              # get the current state
    scores = np.zeros(len(agents))

    for t in range(max_t):
        # select actions
        actions = [agent.act(states[i], add_noise=False) for i, agent in enumerate(agents)]
        env_info = env.step(actions)[brain_name]       # send the action to the environment
        rewards = env_info.rewards                     # get the reward
        dones = env_info.local_done                    # see if episode has finished
        states = env_info.vector_observations          # roll over the state to next time step
        scores += rewards                              # update the score
        if np.any(dones):                              # exit loop if episode finished
            break
    
    print('Score: {:.5f}'.format(np.max(scores)))

# Select environment of Reacher
env = UnityEnvironment(file_name=tennis_filename)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

# Set device between cuda:0 and cpu
torch_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device =', torch_device)

memory = ReplayBuffer(action_size, memory_params['buffer_size'],
                      memory_params['batch_size'], memory_params['seed'], torch_device)

# Test
maddpg_agents = [MADDPGAgent(state_size, action_size, memory, torch_device, params)
                 for _ in range(num_agents)]

for i, agent in enumerate(maddpg_agents):
    agent.load_actor_weights(actor_filename + str(i) + '.pth')
    agent.load_critic_weights(critic_filename + '.pth')

test(maddpg_agents)

env.close()

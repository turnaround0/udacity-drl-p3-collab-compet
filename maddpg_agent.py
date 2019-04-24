import numpy as np
import random
import copy
from collections import namedtuple, deque

from maddpg_model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

from noise import OUNoise


class MADDPGAgent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, memory, device='cpu', params=None):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            memory (obj): Memory buffer to sample
            device (str): device string between cuda:0 and cpu
            params (dict): hyper-parameters
        """
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.step_t = 0
        self.update_every = params['update_every']

        # Set parameters
        self.gamma = params['gamma']
        self.tau = params['tau']
        self.seed = random.seed(params['seed'])

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, params['seed'],
                                 params['actor_units'][0], params['actor_units'][1]).to(device)
        self.actor_target = Actor(state_size, action_size, params['seed'],
                                  params['actor_units'][0], params['actor_units'][1]).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=params['lr_actor'])

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, params['seed'],
                                   params['critic_units'][0], params['critic_units'][1]).to(device)
        self.critic_target = Critic(state_size, action_size, params['seed'],
                                    params['critic_units'][0], params['critic_units'][1]).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=params['lr_critic'], weight_decay=params['weight_decay'])

        # Noise process
        self.noise = OUNoise(action_size, params['seed'], theta=params['noise_theta'], sigma=params['noise_sigma'])

        # Replay memory
        self.memory = memory

    def store_weights(self, filenames):
        """Store weights of Actor/Critic

        Params
        ======
            filenames (list): string of filename to store weights of actor and critic
                              filenames[0] = actor weights
                              filenames[1] = critic weights
        """
        torch.save(self.actor_local.state_dict(), filenames[0])
        torch.save(self.critic_local.state_dict(), filenames[1])

    def load_weights(self, filenames):
        """Load weights of Actor/Critic

        Params
        ======
            filenames (list): string of filename to load weights of actor and critic
                              filenames[0] = actor weights
                              filenames[1] = critic weights
        """
        self.actor_local.load_state_dict(torch.load(filenames[0]))
        self.critic_local.load_state_dict(torch.load(filenames[1]))

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        self.step_t = (self.step_t + 1) % self.update_every

        # Learn, if enough samples are available in memory
        if self.step_t == 0 and len(self.memory) > self.memory.get_batch_size():
            experiences = self.memory.sample()
            self.learn(experiences, self.gamma)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device)
        self.actor_local.eval()
        
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

    @staticmethod
    def soft_update(local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

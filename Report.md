### Learning algorithm
Multi Agent Deep Deterministic Policy Gradient[(MADDPG)](https://arxiv.org/abs/1706.02275) algorithm was used for this training.<br>

### Hyper-parameters
```python
memory_params = {
    'buffer_size': int(1e6),        # replay buffer size
    'batch_size': 128,              # minibatch size
    'seed': 0,                      # Seed to generate random numbers
}
```

```python
params = {
    'gamma': 0.99,                      # discount factor
    'tau': 1e-3,                        # for soft update of target parameters
    'update_every': 1,                  # update parameters per this number
    'lr_actor': 7e-5,                   # learning rate of the Actor
    'lr_critic': 1e-4,                  # learning rate of the Critic
    'seed': 0,                          # Seed to generate random numbers
    'actor_units': [512, 256],          # Number of nodes in hidden layers of the Actor
    'critic_units': [512, 256],         # Number of nodes in hidden layers of the Critic
    'weight_decay': 0,                  # L2 weight decay
    'noise_theta': 0.15,                # Theta of Ornstein-Uhlenbeck process
    'noise_sigma': 0.01,                # Sigma of Ornstein-Uhlenbeck process
}
```


### Model architectures
There are two neural networks as the actor and the critic.

* The Actor
1. First layer: (input: state size, 512)
   Activation function: Leaky ReLU
   Batch normalization: (512)
2. Second layer: (512, 256)
   Activation function: Leaky ReLU
   Batch normalization: (256)
3. Output layer: (256, action size)
   Activation function: Tanh

* The Critic
1. First layer: (input: state size, 512)
   Activation function: Leaky ReLU
   Batch normalization: (512)
2. Second layer: (512 + action size, 256)
   Activation function: Leaky ReLU
3. Output layer: (256, 1)
   Activation function: Linear

### Plot of Rewards
![Plot of rewards](maddpg_plot.png)

### Ideas for Future Work

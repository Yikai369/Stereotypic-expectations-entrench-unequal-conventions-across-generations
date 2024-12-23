"""
An implementation of the Proximal Policy Gradient (PPO) algorithm. 

Adapted from the Github Repository: 
Minimal PyTorch Implementation of Proximal Policy Optimization
https://github.com/nikhilbarhate99/PPO-PyTorch
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical

# Memory 
class RolloutBuffer:
    def __init__(self):
        """Initialize the memory buffer."""
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        """Clear the stored memories."""
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


# Actor-critic network 
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        
        # actor 
        self.actor = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 128),
                        nn.Tanh(),
                        # nn.Dropout(p=0.2),
                        nn.Linear(128, 64),
                        nn.Tanh(),
                        nn.Linear(64, action_dim),
                        nn.Softmax(dim=-1)
                    )
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 128),
                        nn.Tanh(),
                        # nn.Dropout(p=0.2),
                        nn.Linear(128, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
        

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        """Takes an action with the input state."""
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state, action):
        """Evaluate the specified action and state."""
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


# PPO  
class PPO:
    def __init__(self, device, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip):
        self.device = device 
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.model1 = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.model1.actor.parameters(), 'lr': lr_actor},
                        {'params': self.model1.critic.parameters(), 'lr': lr_critic}
                    ])
        self.loss_fn = nn.MSELoss()

    def take_action(self, state):
        """Choose an action based on the observed state."""
        with torch.no_grad():
            state = state.to(self.device)
            action, action_logprob = self.model1.act(state)

        return action, action_logprob

    def training(self, buffer, entropy_coefficient=0.01):
        """Train the model with the memories stored in the buffer."""
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(buffer.rewards), reversed(buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(buffer.logprobs, dim=0)).detach().to(self.device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.model1.evaluate(old_states, old_actions)
            
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
           
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO 
            critic_loss = self.loss_fn(state_values, rewards)
            policy_loss = -torch.min(surr1, surr2) 
            loss = policy_loss + 0.5*critic_loss - entropy_coefficient * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        return loss.mean() 
    
    def save(self, checkpoint_path):
        """Save the current model."""
        torch.save(self.model1.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        """Load the specified model."""
        self.model1.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

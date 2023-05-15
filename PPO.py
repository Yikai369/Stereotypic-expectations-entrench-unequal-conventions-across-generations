import torch
import torch.nn as nn
from torch.distributions import Categorical

# Memory class 
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.n_sell = [0,0]
        self.n_buy = [0,0]
        self.day_rewards = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


# class Network 
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
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state, action):

        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


# Class Model  
class PPO:
    def __init__(self, device, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip):

        self.device = device 
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        # self.buffer = RolloutBuffer()

        self.model1 = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.model1.actor.parameters(), 'lr': lr_actor},
                        {'params': self.model1.critic.parameters(), 'lr': lr_critic}
                    ])

        # self.policy_old = ActorCritic(state_dim, action_dim).to(self.device)
        # self.policy_old.load_state_dict(self.model1.state_dict())
        
        self.loss_fn = nn.MSELoss()

    def take_action(self, state):
        with torch.no_grad():
            state = state.to(self.device)
            # action, action_logprob = self.policy_old.act(state)
            action, action_logprob = self.model1.act(state)

        # buffer.states.append(state)
        # buffer.actions.append(action)
        # buffer.logprobs.append(action_logprob)

        return action, action_logprob

    def training(self, buffer, entropy_coefficient=0.01):
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
        # print('state:', old_states.shape)
        old_actions = torch.squeeze(torch.stack(buffer.actions, dim=0)).detach().to(self.device)
        # print('state:', old_actions.shape)
        old_logprobs = torch.squeeze(torch.stack(buffer.logprobs, dim=0)).detach().to(self.device)
        # print('state:', old_logprobs.shape)


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
            # print('c_loss:', critic_loss)
            policy_loss = -torch.min(surr1, surr2) 
            # print('p-loss', dist_entropy.shape)
            loss = policy_loss + 0.5*critic_loss - entropy_coefficient * dist_entropy
            # loss = policy_loss + 0.5*critic_loss

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        # self.policy_old.load_state_dict(self.model1.state_dict())

        # clear buffer
        # self.buffer.clear()

        return loss.mean() 
    
    def save(self, checkpoint_path):
        # torch.save(self.policy_old.state_dict(), checkpoint_path)
        torch.save(self.model1.state_dict(), checkpoint_path)

   
    def load(self, checkpoint_path):
        # self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.model1.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

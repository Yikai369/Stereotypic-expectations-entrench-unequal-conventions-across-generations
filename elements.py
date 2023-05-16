import torch
import random
from collections import deque

from env import generate_input


class Agent():
    kind = "agent"  # class variable shared by all instances

    def __init__(self, model, agent_type, appearance, 
                 wood_skill, stone_skill, house_skill):
        """Initialize the agent entity."""
        self.appearance = appearance  
        self.policy = model  
        self.reward = 0  
        self.episode_memory = None 
        self.has_transitions = True
        self.action_type = "neural_network"
        self.wood = 0
        self.stone = 0
        self.house = 0
        self.wood_skill = wood_skill
        self.stone_skill = stone_skill
        self.house_skill = house_skill
        self.coin = 6
        self.agent_type = agent_type
        self.state = torch.zeros(6).float()


    def transition(self, env, models, action, done, 
                   agent_list, agent, pred_success):
        """Transit the environment and the agent to 
        the next state based on the selected action.
        """
        reward = 0

        # Chop wood 
        if action == 0:
            if random.random() < self.wood_skill and ((self.wood + self.stone) < 11):
                self.wood = self.wood + 1
                reward = 0

        # Mine stones
        if action == 1:
            if random.random() < self.stone_skill and ((self.wood + self.stone) < 11):
                self.stone = self.stone + 1
                reward = 0

        # Build houses
        if action == 2:
            dice_role = random.random()
            if dice_role < self.house_skill and self.wood > 0 and self.stone > 0:
                self.wood = self.wood - 1
                self.stone = self.stone - 1
                self.house = self.house + 1
                self.coin = self.coin + 15
                reward = 15

        # Sell wood
        if action == 3:
            if self.wood > 1 and pred_success:
                self.wood = self.wood - 2
                reward = 1
                self.coin = self.coin + 1
                env.wood = env.wood + 2

        # Sell stone
        if action == 4:
            if self.stone > 1 and pred_success:
                self.stone = self.stone - 2
                reward = 1
                self.coin = self.coin + 1
                env.stone = env.stone + 2

        #  Buy wood
        if action == 5:
            if env.wood > 1 and self.coin > 1:
                env.wood = env.wood - 2
                reward = -2
                self.coin = self.coin - 2
                self.wood = self.wood + 2
        
        # Buy stones  
        if action == 6:
            if env.stone > 1 and self.coin > 1:
                env.stone = env.stone - 2
                reward = -2
                self.coin = self.coin - 2
                self.stone = self.stone + 2

        # Generate the next state 
        next_state, _ = generate_input(agent_list, agent, agent_list[agent].state)
        next_state = next_state.unsqueeze(0).to(models[0].device)

        return env, reward, next_state, done

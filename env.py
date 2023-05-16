"""An implementation of the AIEconimist Environment."""

import torch

class AIEcon_simple_game:
    def __init__(self):
        """Initialize the game."""
        self.wood = 6
        self.stone = 6
        self.action_space = 7


def generate_input(agent_list, agent, state):
    """Generate observed states for agents residing in the game environment."""
    previous_state = state

    # continuous values measuring the agent's resource possessions 
    cur_wood = agent_list[agent].wood /5 
    cur_stone = agent_list[agent].stone /5
    cur_coin = agent_list[agent].coin /5

    # binary values measuring the agent's resource possessions 
    suf_wood = 0    
    suf_stone = 0
    suf_coin = 0
    if agent_list[agent].wood > 1: 
        suf_wood = 1
    else:
        suf_wood = 0
    if agent_list[agent].stone > 1:
        suf_stone = 1
    else:
        suf_stone = 0
    if agent_list[agent].coin >0:
        suf_coin = 1
    else:
        suf_coin = 0

    # Full observed state 
    state = torch.tensor([cur_wood, cur_stone, cur_coin, suf_wood, suf_stone, suf_coin]).float()

    return state, previous_state
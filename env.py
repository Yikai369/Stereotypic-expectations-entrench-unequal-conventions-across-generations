import torch

class AIEcon_simple_game:
    def __init__(self):
        self.wood = 6
        self.stone = 6

    def pov(self, world, location, holdObject, inventory=[]):
        """
        TODO: rewrite pov to simply take in a vector
        """

        previous_state = holdObject.episode_memory[-1][1][0]
        current_state = previous_state.clone()

        current_state[:, 0:-1, :] = previous_state[:, 1:, :]


        current_state[:, -1, :] = state

        return current_state


def generate_input(agent_list, agent, state):

    previous_state = state


    cur_wood = agent_list[agent].wood /5 # what is the best way of having current inventory
    cur_stone = agent_list[agent].stone /5
    cur_coin = agent_list[agent].coin /5

    suf_wood = 0    
    suf_stone = 0
    suf_coin = 0
    if agent_list[agent].wood > 1: # does it make sense to also have a degree of whether it can buy or sell as a binary?
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
    state = torch.tensor([cur_wood, cur_stone, cur_coin, suf_wood, suf_stone, suf_coin]).float()


    return state, previous_state

def prepare_lstm(agent_list, agent, state):
    previous_state = agent_list[agent].episode_memory[-1][1][0]
    current_state = previous_state.clone()
    current_state[:, 0:-1, :] = previous_state[:, 1:, :]
    current_state[:, -1, :] = state
    return current_state

def prepare_lstm2(previous_state, next_state):

    current_state = previous_state.clone()
    current_state[:, 0:-1, :] = previous_state[:, 1:, :]
    current_state[:, -1, :] = next_state

    return current_state

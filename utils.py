# import modules
import random
import numpy as np
import torch
import pandas as pd
from collections import deque

from elements import Agent
from env import AIEcon_simple_game, generate_input
from PPO import RolloutBuffer, PPO

# helper functions
def unit_test(decider_model, agent_list, unseen_agents): 
    unseen_appearances = [torch.tensor(val).float() for val in unseen_agents]
    all_agent_appearances = [torch.tensor(agent.appearance).float() for agent in agent_list]
    all_agent_appearances.extend(unseen_appearances)
    agent_appearances = torch.stack(all_agent_appearances) 
    decisions, action_logprobs = decider_model.take_action(agent_appearances) 
    probs = np.exp(action_logprobs)

    return decisions, probs 


def replace_agents_equal_ratio(agent_list, population_size, attributes, prop, prop_reverse, replace_prop, iteration):
    count = int(iteration * replace_prop * population_size)  
    for count1, agent_type in enumerate([0,1,2]): 
        for count2, agent_subtype in enumerate([0,1,2]): 
            list_subtype_agent = []
            if agent_type == 0: 
                group_id = [1,0,0]
                if agent_subtype == 0: 
                    list_subtype_agent = [Agent(0,0,group_id+attributes[count+i], .95, .15, .05) 
                                          for i in range(int(replace_prop*(population_size/3)*prop))]
                    count += int(replace_prop*(population_size/3)*prop) 
                elif agent_subtype == 1:
                    list_subtype_agent = [Agent(1,1,group_id+attributes[count+i], .15, .95, .05) 
                                          for i in range(int(replace_prop*(population_size/3)*prop_reverse))]
                    count += int(replace_prop*(population_size/3)*prop_reverse)
                elif agent_subtype == 2:
                    list_subtype_agent = [Agent(2,2,group_id+attributes[count+i], .1, .1, .95) 
                                          for i in range(int(replace_prop*(population_size/3)*0.1))]
                    count += int(replace_prop*(population_size/3)*0.1)

            elif agent_type == 1:
                group_id = [0,1,0]
                if agent_subtype == 0: 
                    list_subtype_agent = [Agent(3,3,group_id+attributes[count+i], .95, .15, .05) 
                                          for i in range(int(replace_prop*(population_size/3)*prop_reverse))]
                    count += int(replace_prop*(population_size/3)*prop_reverse)
                elif agent_subtype == 1:
                    list_subtype_agent = [Agent(4,4,group_id+attributes[count+i], .15, .95, .05) 
                                          for i in range(int(replace_prop*(population_size/3)*prop))]
                    count += int(replace_prop*(population_size/3)*prop)
                elif agent_subtype == 2:
                    list_subtype_agent = [Agent(5,5,group_id+attributes[count+i], .1, .1, .95) 
                                          for i in range(int(replace_prop*(population_size/3)*0.1))]
                    count += int(replace_prop*(population_size/3)*0.1)

            elif agent_type == 2:
                group_id = [0,0,1]
                if agent_subtype == 0: 
                    list_subtype_agent = [Agent(6,6,group_id+attributes[count+i], .95, .15, .05) 
                                          for i in range(int(replace_prop*(population_size/3)*0.1))]
                    count += int(replace_prop*(population_size/3)*0.1)
                elif agent_subtype == 1:
                    list_subtype_agent = [Agent(7,7,group_id+attributes[count+i], .15, .95, .05) 
                                          for i in range(int(replace_prop*(population_size/3)*0.1))]
                    count += int(replace_prop*(population_size/3)*0.1)
                elif agent_subtype == 2: 
                    
                    list_subtype_agent = [Agent(8,8,group_id+attributes[count+i], .1, .1, .95) 
                                          for i in range(int((replace_prop*population_size/3)*0.8))]
                    count += int(replace_prop*(population_size/3)*0.8)

            for agent in list_subtype_agent:
                agent.episode_memory = RolloutBuffer()
            agent_list.extend(list_subtype_agent)


def create_models(N_model, device):
    models = deque(maxlen=N_model)
    for i in range(N_model):
        models.append(
            PPO(
                device=device, 
                state_dim=6,
                action_dim=7,
                lr_actor=0.001,
                lr_critic=0.0005,
                gamma=0.9,
                K_epochs=10,
                eps_clip=0.2 
            )
        ) 

    # convert to device
    for model in range(len(models)):
        models[model].model1.to(device)

    return models


def create_agents(population_size, agent_distribution, original_agents_attributes):
    agent_list = deque(maxlen=population_size)
    count = 0
    for count1, agent_type in enumerate([0,1,2]):
        for count2, agent_subtype in enumerate([0,1,2]):
            list_subtype_agent = []
          
            if agent_type == 0:
                group_id = [1,0,0]
                if agent_subtype == 0:
                    list_subtype_agent = [Agent(0,0,group_id+original_agents_attributes[count+i], .95, .15, .05)
                                          for i in range(agent_distribution['wood'][0])]
                    count += agent_distribution['wood'][0]
                elif agent_subtype == 1:
                    list_subtype_agent = [Agent(1,1,group_id+original_agents_attributes[count+i], .15, .95, .05)
                                          for i in range(agent_distribution['wood'][1])]
                    count += agent_distribution['wood'][1]
                elif agent_subtype == 2:
                    list_subtype_agent = [Agent(2,2,group_id+original_agents_attributes[count+i], .1, .1, .95)
                                          for i in range(agent_distribution['wood'][2])]
                                          
                    count += agent_distribution['wood'][2]
                    
            elif agent_type == 1:
                group_id = [0,1,0]
                if agent_subtype == 0:
                    list_subtype_agent = [Agent(3,3,group_id+original_agents_attributes[count+i], .95, .15, .05)
                                          for i in range(agent_distribution['stone'][0])]
                    count += agent_distribution['stone'][0]
                elif agent_subtype == 1:
                    list_subtype_agent = [Agent(4,4,group_id+original_agents_attributes[count+i], .15, .95, .05)
                                          for i in range(agent_distribution['stone'][1])]
                    count += agent_distribution['stone'][1]
                elif agent_subtype == 2:
                    list_subtype_agent = [Agent(5,5,group_id+original_agents_attributes[count+i], .1, .1, .95)
                                          for i in range(agent_distribution['stone'][2])]
                    count += agent_distribution['stone'][2]
                    
            elif agent_type == 2:
                group_id = [0,0,1]
                if agent_subtype == 0:
                    list_subtype_agent = [Agent(6,6,group_id+original_agents_attributes[count+i], .95, .15, .05)
                                          for i in range(agent_distribution['house'][0])]
                    count += agent_distribution['house'][0]
                elif agent_subtype == 1:
                    list_subtype_agent = [Agent(7,7,group_id+original_agents_attributes[count+i], .15, .95, .05)
                                          for i in range(agent_distribution['house'][1])]
                    count += agent_distribution['house'][1]
                elif agent_subtype == 2:
                    list_subtype_agent = [Agent(8,8,group_id+original_agents_attributes[count+i], .1, .1, .95)
                                          for i in range(agent_distribution['house'][2])]
                    count += agent_distribution['house'][2]

            for agent in list_subtype_agent:
                agent.episode_memory = RolloutBuffer()
            agent_list.extend(list_subtype_agent)
      
    random.shuffle(agent_list)
    return agent_list

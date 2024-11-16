"""Utility functions."""
import random
import numpy as np
import torch
import itertools
from collections import deque

from elements import Agent
from PPO import RolloutBuffer, PPO


def create_identity_feature(
        study_number,
        length, 
        population_size,
        n_unseen_agents
        ): 
    """Create identity code with a specified length for agents.
    
    Args: 
        study_number: The index of the study. 
        length: The length of the identity code, not including the common 
            feature ([0,0,0]) and the group feature. 
        population_size: The total number of agents in the game. 
        n_unseen_agents: The number of novel agents the market faces in 
            each unit test. 
    """
    binary = [0,1]
    all_individual_attributes = [list(val) for val in list(itertools.product(binary, repeat=length))]
    random.shuffle(all_individual_attributes)

    original_agents_attributes, unseen_attributes = all_individual_attributes[:population_size], \
                                                    all_individual_attributes[population_size:population_size*2]
    original_agents_attributes = [list(tuple(i)+(0, 0, 0)) for i in original_agents_attributes]
    replacement_attributes = [list(tuple(i)+(0, 0, 0)) for i in unseen_attributes]
    
    # If study number is 3, generate novel features for unit tests 
    unseen_agent_groups = None 
    unseen_agents_attributes = None 
    if study_number == 3: 
        unseen_agents_attributes = random.sample(
            all_individual_attributes[population_size*2:], 
            n_unseen_agents
            )
        unseen_agents_attributes = [list(tuple(i) + (0, 0, 0)) for i in unseen_agents_attributes] 
        # Add group-characterized feature to the identity code 
        unseen_agents_attributes = [[1,0,0]+val for val in unseen_agents_attributes[:40]] \
                        + [[0,1,0]+val for val in unseen_agents_attributes[40:80]] \
                        + [[0,0,1] +val for val in unseen_agents_attributes[80:]]
        print('number of unseen agents:', len(unseen_agents_attributes))
        # Group membership of each unseen agent 
        unseen_agent_groups = [ind//(n_unseen_agents/3) for ind in range(len(unseen_agents_attributes))]

    return original_agents_attributes, replacement_attributes, unseen_agents_attributes, unseen_agent_groups


def create_agents(
        population_size, 
        agent_distribution, 
        original_agents_attributes, 
        study_number
        ):
    """Create agents of three groups (wood, stone, house), each consisting of 
    three subgroups (chopping, mining, building specialists).

    Args:
        population_size: The total number of agents in the game.
        agent_distribution: The number of different types of agents in each group. 
        original_agents_attributes: A list containing all the identity features for
            the original agents in the game. 
        study_number: The index of the study. 
    """
    agent_list = deque(maxlen=population_size)
    count = 0

    for agent_type in [0,1,2]:

        for agent_subtype in [0,1,2]:
            list_subtype_agent = []

            # Wood group
            if agent_type == 0:
                group_id = [1,0,0]
                if study_number == 1: 
                    group_id = [] 
                # Chopping specialists
                if agent_subtype == 0:
                    list_subtype_agent = [Agent(0,0,group_id+original_agents_attributes[count+i], .95, .15, .05)
                                          for i in range(agent_distribution['wood'][0])]
                    count += agent_distribution['wood'][0]
                # Mining specialists
                elif agent_subtype == 1:
                    list_subtype_agent = [Agent(1,1,group_id+original_agents_attributes[count+i], .15, .95, .05)
                                          for i in range(agent_distribution['wood'][1])]
                    count += agent_distribution['wood'][1]
                # Building specialists
                elif agent_subtype == 2:
                    list_subtype_agent = [Agent(2,2,group_id+original_agents_attributes[count+i], .1, .1, .95)
                                          for i in range(agent_distribution['wood'][2])]
                                          
                    count += agent_distribution['wood'][2]
            
            # Stone group 
            elif agent_type == 1:
                group_id = [0,1,0]
                if study_number == 1: 
                    group_id = []
                # Chopping specialists
                if agent_subtype == 0:
                    list_subtype_agent = [Agent(3,3,group_id+original_agents_attributes[count+i], .95, .15, .05)
                                          for i in range(agent_distribution['stone'][0])]
                    count += agent_distribution['stone'][0]
                # Mining specialists 
                elif agent_subtype == 1:
                    list_subtype_agent = [Agent(4,4,group_id+original_agents_attributes[count+i], .15, .95, .05)
                                          for i in range(agent_distribution['stone'][1])]
                    count += agent_distribution['stone'][1]
                # Building specialists
                elif agent_subtype == 2:
                    list_subtype_agent = [Agent(5,5,group_id+original_agents_attributes[count+i], .1, .1, .95)
                                          for i in range(agent_distribution['stone'][2])]
                    count += agent_distribution['stone'][2]
            
            # House group 
            elif agent_type == 2:
                group_id = [0,0,1]
                if study_number == 1: 
                    group_id = []
                # Chopping specialists
                if agent_subtype == 0:
                    list_subtype_agent = [Agent(6,6,group_id+original_agents_attributes[count+i], .95, .15, .05)
                                          for i in range(agent_distribution['house'][0])]
                    count += agent_distribution['house'][0]
                # Mining specialists
                elif agent_subtype == 1:
                    list_subtype_agent = [Agent(7,7,group_id+original_agents_attributes[count+i], .15, .95, .05)
                                          for i in range(agent_distribution['house'][1])]
                    count += agent_distribution['house'][1]
                # Building specialists
                elif agent_subtype == 2:
                    list_subtype_agent = [Agent(8,8,group_id+original_agents_attributes[count+i], .1, .1, .95)
                                          for i in range(agent_distribution['house'][2])]
                    count += agent_distribution['house'][2]

            # Create memory buffers for every agent 
            for agent in list_subtype_agent:
                agent.episode_memory = RolloutBuffer()

            agent_list.extend(list_subtype_agent)
      
    random.shuffle(agent_list)
    return agent_list


def replace_agents_equal_ratio(
        agent_list, 
        population_size, 
        attributes, 
        prop, 
        prop_reverse, 
        replace_prop, 
        iteration
        ):
    """Replace the old population with a group of new agents that involves 
    equal quantities of chopping and mining specialists.

    Args: 
        agent_list: A list containing all the agent entities. 
        attributes: Identity features of the new agents. 
        prop: The proportion of majority agents in the new population 
            entering the game. 
        prop_reverse: The proportion of minority agents in the new population 
            entering the game. 
        repalce_prop: The proportion of agents replaced every iteration 
        iteration: The index of the replacement 
    """
    count = int(iteration * replace_prop * population_size)  

    for agent_type in [0,1,2]: 

        for agent_subtype in [0,1,2]: 
            list_subtype_agent = []

            # Wood group 
            if agent_type == 0: 
                group_id = [1,0,0]
                # Chopping specialists
                if agent_subtype == 0: 
                    list_subtype_agent = [Agent(0,0,group_id+attributes[count+i], .95, .15, .05) 
                                          for i in range(int(replace_prop*(population_size/3)*prop))]
                    count += int(replace_prop*(population_size/3)*prop) 
                # Mining specialists
                elif agent_subtype == 1:
                    list_subtype_agent = [Agent(1,1,group_id+attributes[count+i], .15, .95, .05) 
                                          for i in range(int(replace_prop*(population_size/3)*prop_reverse))]
                    count += int(replace_prop*(population_size/3)*prop_reverse)
                # Building specialists
                elif agent_subtype == 2:
                    list_subtype_agent = [Agent(2,2,group_id+attributes[count+i], .1, .1, .95) 
                                          for i in range(int(replace_prop*(population_size/3)*0.1))]
                    count += int(replace_prop*(population_size/3)*0.1)

            # Stone group
            elif agent_type == 1:
                group_id = [0,1,0]
                # Chopping specialists
                if agent_subtype == 0: 
                    list_subtype_agent = [Agent(3,3,group_id+attributes[count+i], .95, .15, .05) 
                                          for i in range(int(replace_prop*(population_size/3)*prop_reverse))]
                    count += int(replace_prop*(population_size/3)*prop_reverse)
                # Mining specialists
                elif agent_subtype == 1:
                    list_subtype_agent = [Agent(4,4,group_id+attributes[count+i], .15, .95, .05) 
                                          for i in range(int(replace_prop*(population_size/3)*prop))]
                    count += int(replace_prop*(population_size/3)*prop)
                # Building specialists
                elif agent_subtype == 2:
                    list_subtype_agent = [Agent(5,5,group_id+attributes[count+i], .1, .1, .95) 
                                          for i in range(int(replace_prop*(population_size/3)*0.1))]
                    count += int(replace_prop*(population_size/3)*0.1)

            # House group 
            elif agent_type == 2:
                group_id = [0,0,1]
                # Chopping specialists
                if agent_subtype == 0: 
                    list_subtype_agent = [Agent(6,6,group_id+attributes[count+i], .95, .15, .05) 
                                          for i in range(int(replace_prop*(population_size/3)*0.1))]
                    count += int(replace_prop*(population_size/3)*0.1)
                # Mining specialists
                elif agent_subtype == 1:
                    list_subtype_agent = [Agent(7,7,group_id+attributes[count+i], .15, .95, .05) 
                                          for i in range(int(replace_prop*(population_size/3)*0.1))]
                    count += int(replace_prop*(population_size/3)*0.1)
                # Building specialists
                elif agent_subtype == 2: 
                    list_subtype_agent = [Agent(8,8,group_id+attributes[count+i], .1, .1, .95) 
                                          for i in range(int((replace_prop*population_size/3)*0.8))]
                    count += int(replace_prop*(population_size/3)*0.8)

            # Create memory buffers for every agent 
            for agent in list_subtype_agent:
                agent.episode_memory = RolloutBuffer()

            agent_list.extend(list_subtype_agent)


def create_models(
        num_model, 
        device,
        is_a2c=False
        ):
    """Create models for agents."""
    models = deque(maxlen=num_model)
    for _ in range(num_model):
        models.append(
            PPO(
                device=device, 
                state_dim=6,
                action_dim=7,
                lr_actor=0.001,
                lr_critic=0.0005,
                gamma=0.9,
                K_epochs=10,
                eps_clip=0.2,
                is_a2c=is_a2c 
            )
        ) 
    # convert to device
    for model in range(len(models)):
        models[model].model1.to(device)
    return models


def unit_test(
        decider_model, 
        agent_list, 
        unseen_agents
        ): 
    """Test the market's expectations towards novel agents.
    
    Args: 
        decider_model: The market decider's model. 
        agent_list: A list containing all the agent entities. 
        unseen_agents: A list containing all unseen agents' identity code. 
    """
    unseen_appearances = [torch.tensor(val).float() for val in unseen_agents]
    all_agent_appearances = [torch.tensor(agent.appearance).float() for agent in agent_list]
    all_agent_appearances.extend(unseen_appearances)
    agent_appearances = torch.stack(all_agent_appearances) 

    # Obtain the market decider's expectations/predictions towards the seen and unseen agents
    decisions, action_logprobs = decider_model.take_action(agent_appearances) 
    probs = np.exp(action_logprobs)

    return decisions, probs 


def generate_agent_distribution(
        prop, 
        prop_reverse, 
        population_size
        ):
    """Generate the specific agent distribution for populations
    with different sizes.

    Args:
        prop: The proportion of majority agents in the new population 
            entering the game. 
        prop_reverse: The proportion of minority agents in the new population 
            entering the game. 
    """
    if population_size in (300, 600):
        n_agents_per_group = population_size / 3
        wood_group_distribution = [int(n_agents_per_group*prop), 
                                   int(n_agents_per_group*prop_reverse), 
                                   int(n_agents_per_group*(1-prop-prop_reverse))]
        stone_group_distribution = [int(n_agents_per_group*prop_reverse), 
                                    int(n_agents_per_group*prop), 
                                    int(n_agents_per_group*(1-prop-prop_reverse))]
        house_group_distribution = [int(n_agents_per_group*0.1), 
                                    int(n_agents_per_group*0.1), 
                                    int(n_agents_per_group*0.8)]
        agent_distribution = {'wood': wood_group_distribution, 
                              'stone': stone_group_distribution, 
                              'house': house_group_distribution}
    elif population_size == 30:
        agent_distribution = {'wood': [6, 3, 1], 
                              'stone': [3, 6, 1], 
                              'house': [1, 1, 8]}
    elif population_size == 100:
        agent_distribution = {'wood': [16, 8, 8], 
                              'stone': [8, 16, 8], 
                              'house':[4, 4, 28]}
    else: 
        raise ValueError('Population size should be 30, 100, 300, or 600.')
    
    return agent_distribution


def save_data(
        save_dir, 
        study_prefix,
        population_size, 
        prop, 
        run, 
        all_agent_actions_sum,
        total_normalized_reward_sum, 
        policy_normalized_reward_sum,
        decider_reward_sum, 
        interaction_record_sum,
        iteration=None
        ): 
    """Save simulation data.
    
    Args: 
        prop: The proportion of majority agents in the new population 
            entering the game. 
        all_agent_actions_sum: Sum action frequencies of each subtype of agents 
            in one run of a condition. 
        total_normalized_reward_sum: Normalized collective returns in one run 
            of a condition.
        policy_normalized_reward_sum: Normalized subgroup returns in one run 
            of a condition. 
        decider_reward_sum: Sum rewards of the market decider in one run
            of a condition. 
        interation_record_sum: The history of the market's predictions 
            towards agents during transactions. 
    """
    suffix = f'_{iteration}' if iteration is not None else '' 
    np.savetxt(save_dir+f"study{study_prefix}_{population_size}_{prop}_{run}"+suffix+"_agent1_actions.csv", 
                np.array(all_agent_actions_sum[0]), delimiter=',', fmt='%s')
    np.savetxt(save_dir+f"study{study_prefix}_{population_size}_{prop}_{run}"+suffix+"_agent2_actions.csv", 
                np.array(all_agent_actions_sum[1]), delimiter=',', fmt='%s')
    np.savetxt(save_dir+f"study{study_prefix}_{population_size}_{prop}_{run}"+suffix+"_agent3_actions.csv",
                np.array(all_agent_actions_sum[2]), delimiter=',', fmt='%s')
    np.savetxt(save_dir+f"study{study_prefix}_{population_size}_{prop}_{run}"+suffix+"_agent4_actions.csv", 
                np.array(all_agent_actions_sum[3]), delimiter=',', fmt='%s')
    np.savetxt(save_dir+f"study{study_prefix}_{population_size}_{prop}_{run}"+suffix+"_agent5_actions.csv", 
                np.array(all_agent_actions_sum[4]), delimiter=',', fmt='%s')
    np.savetxt(save_dir+f"study{study_prefix}_{population_size}_{prop}_{run}"+suffix+"_agent6_actions.csv", 
                np.array(all_agent_actions_sum[5]), delimiter=',', fmt='%s')
    np.savetxt(save_dir+f"study{study_prefix}_{population_size}_{prop}_{run}"+suffix+"_agent7_actions.csv", 
                np.array(all_agent_actions_sum[6]), delimiter=',', fmt='%s')
    np.savetxt(save_dir+f"study{study_prefix}_{population_size}_{prop}_{run}"+suffix+"_agent8_actions.csv", 
                np.array(all_agent_actions_sum[7]), delimiter=',', fmt='%s')
    np.savetxt(save_dir+f"study{study_prefix}_{population_size}_{prop}_{run}"+suffix+"_agent9_actions.csv", 
                np.array(all_agent_actions_sum[8]), delimiter=',', fmt='%s')

    np.savetxt(save_dir+f"study{study_prefix}_{population_size}_{prop}_{run}"+suffix+"_total_normalized_agent_rewards.csv", 
                np.array(total_normalized_reward_sum),  delimiter=',', fmt='%s') 
    np.savetxt(save_dir+f"study{study_prefix}_{population_size}_{prop}_{run}"+suffix+"_policy_normalized_agent_rewards.csv", 
                np.array(policy_normalized_reward_sum), delimiter=',', fmt='%s')
    np.savetxt(save_dir+f"study{study_prefix}_{population_size}_{prop}_{run}"+suffix+"_decider_rewards.csv", 
                np.array(decider_reward_sum), delimiter=',', fmt='%s')
    np.savetxt(save_dir+f"study{study_prefix}_{population_size}_{prop}_{run}"+suffix+"_interaction_record.csv", 
                np.array(interaction_record_sum), delimiter=',', fmt='%s')
    

def show_training_progress(
        iteration,
        epoch, 
        population_size, 
        prop, 
        run, 
        losses, 
        decider_losses,
        rewards, 
        num_agent_per_policy, 
        number_agent_of_interest,
        interaction_record, 
        all_agent_actions
        ):
    """Print the training progress."""
    print("--------------------------------------")
    subgroup_reward = [round(a/b, 2) for a,b in zip(rewards,num_agent_per_policy)]
    total_reward = sum([rewards[i] for i in [0,1,3,4]])/number_agent_of_interest
    print("condition:", population_size, prop)
    print("number of agents in each subgroup:", num_agent_per_policy)
    print("run:", run, "iteration:", iteration, "epoch:" , epoch, 
          "loss: ",losses, "decider loss: ", decider_losses, 
          "\n", "points (wood, stone, house): ", subgroup_reward)
    print('normalized benefits of the microsociety:', total_reward)
    print("chop, mine, build, sell_wood, sell_stone, buy_wood, buy_stone")
    print("agent1 behaviours - chop_c: ", all_agent_actions[0])
    print("agent2 behaviours - chop_m: ", all_agent_actions[1])
    print("agent3 behaviours - chop_h: ", all_agent_actions[2])
    print("agent4 behaviours - mine_c: ", all_agent_actions[3])
    print("agent5 behaviours - mine_m: ", all_agent_actions[4])
    print("agent6 behaviours - mine_h: ", all_agent_actions[5])
    print("agent7 behaviours - hous_c: ", all_agent_actions[6])
    print("agent8 behaviours - hous_m: ", all_agent_actions[7])
    print("agent9 behaviours - hous_h: ", all_agent_actions[8])
    print("Guess wood -- majority of wood group:", interaction_record[0], 
            "  Guess stone -- majority of wood group:", interaction_record[1], ' ', 
            interaction_record[0]/(interaction_record[0]+interaction_record[1]+1e-7))
    print("Guess wood -- minority of wood group:", interaction_record[2], 
            "  Guess stone -- minority of wood group:", interaction_record[3], ' ', 
            interaction_record[2]/(interaction_record[2]+interaction_record[3]+1e-7))
    print("Guess wood -- majority of stone group:", interaction_record[4], 
            "  Guess stone -- majority of stone group:", interaction_record[5], ' ', 
            interaction_record[5]/(interaction_record[4]+interaction_record[5]+1e-7))
    print("Guess wood -- minority of stone group:", interaction_record[6], 
            "  Guess stone -- minority of stone group:", interaction_record[7], ' ', 
            interaction_record[7]/(interaction_record[6]+interaction_record[7]+1e-7))
    
    return subgroup_reward, total_reward


def update_history(
        subgroup_reward, 
        total_reward,
        all_agent_actions_sum, 
        policy_normalized_reward_sum,
        total_normalized_reward_sum, 
        decider_reward_sum, 
        interaction_record_sum, 
        history):
    """Update the history of one condition simulation. """
    for i in range(9): 
        all_agent_actions_sum[i].append(history[5][i])

    policy_normalized_reward_sum.append(subgroup_reward)
    total_normalized_reward_sum.append(total_reward)
    decider_reward_sum.append(history[1])
    interaction_record_sum.append(history[4])




import random
import numpy as np
import torch
import pandas as pd
from collections import deque

from elements import Agent
from env import AIEcon_simple_game, generate_input
from PPO import RolloutBuffer, PPO
from utils import create_models, create_agents, replace_agents_equal_ratio, unit_test


# Set directory
model_path = 'save/'
data_path = 'save/'
unit_test_path = 'save/'
dir_group_data = 'save/'

# Training and environment hyperparameters
device = "cpu"
print(device)
num_epoch = 2
num_run = 30
max_turns = 50
population_size = 300
prop = 0.5
prop_reverse = 0.75-prop
replace_prop = 0.2
unit_test_frequency = 1 
n_unseen_agents = 120

n_agents_per_group = population_size/3
wood_group_distribution = [int(n_agents_per_group*prop), int(n_agents_per_group*prop_reverse), int(n_agents_per_group*(1-prop-prop_reverse))]
stone_group_distribution = [int(n_agents_per_group*prop_reverse), int(n_agents_per_group*prop), int(n_agents_per_group*(1-prop-prop_reverse))]
house_group_distribution = [int(n_agents_per_group*0.1), int(n_agents_per_group*0.1), int(n_agents_per_group*0.8)]
agent_distribution = {'wood': wood_group_distribution, 'stone': stone_group_distribution, 'house': house_group_distribution}

# Create dataframes for data collection
agent_actions = pd.DataFrame(columns=['run', 'iteration', 'epoch']+[f"Agent{i+1}"for i in range(population_size*2)])
agent_rewards = pd.DataFrame(columns=['run', 'iteration', 'epoch']+[f"Agent{i+1}"for i in range(population_size*2)])
agent_property = pd.DataFrame(columns=['agent_type', 'number_iteration'])

# Main loop
for run in range(num_run):
    decider_model = PPO(
                device=device, 
                state_dim=16,
                action_dim=2,
                lr_actor=0.001,
                lr_critic=0.0005,
                gamma=0.9,
                K_epochs=10,
                eps_clip=0.2 
            )
    
    decider_model.replay = RolloutBuffer()

    decider_model.model1.to(device)

    models = create_models(population_size, device)

    env = AIEcon_simple_game()

    i11, i12, i13 = 0, 0, 0


    binary = [0,1]
    
    # Create the attributes for all the agents that will occur in the environment
    appearence_all = [] 
    num_agent_per_policy = [0 for i in range(9)]
    all_individual_attributes = [[i1, i2, i3, i4, i5, i6, i7, i8, i9, i10]
                                 for i1 in binary for i2 in binary for i3 in binary 
                                 for i4 in binary for i5 in binary for i6 in binary 
                                 for i7 in binary for i8 in binary for i9 in binary 
                                 for i10 in binary]

    random.shuffle(all_individual_attributes)
    
    original_agents_attributes, replacement_attributes = all_individual_attributes[:population_size], all_individual_attributes[population_size:population_size*2]
    original_agents_attributes = [list(tuple(i)+(i11, i12, i13)) for i in original_agents_attributes]
    replacement_attributes = [list(tuple(i)+(i11, i12, i13)) for i in replacement_attributes]
    unseen_agents = random.sample(all_individual_attributes[population_size*2:], n_unseen_agents)
    unseen_agents = [list(tuple(i) + (i11, i12, i13)) for i in unseen_agents] 
    unseen_agents = [[1,0,0]+val for val in unseen_agents[:40]] + [[0,1,0]+val for val in unseen_agents[40:80]] + [[0,0,1] +val for val in unseen_agents[80:]]
    print('number of unseen agents:', len(unseen_agents))
    unseen_agent_groups = [ind//(n_unseen_agents/3) for ind in range(len(unseen_agents))]

    # Create the initial agents
    agent_list = create_agents(population_size, agent_distribution, original_agents_attributes)

    num_agents = len(agent_list)
    print(num_agents)

    for i in range(9): 
        num_agent_per_policy[i] = len([agent for agent in agent_list if agent.policy==i]) 
    number_agent_of_interest = 0 
    for i in range(9): 
        if i in [0,1,3,4]:
            number_agent_of_interest += num_agent_per_policy[i] 
    print(number_agent_of_interest)
    print(num_agent_per_policy)

    rewards = [0,0,0,0,0,0,0,0,0]
    decider_rewards = 0 
    losses = 0
    decider_losses = 0

    trainable_models = [i for i in range(population_size)]
    agent1_actions = [0,0,0,0,0,0,0]
    agent2_actions = [0,0,0,0,0,0,0]
    agent3_actions = [0,0,0,0,0,0,0]
    agent4_actions = [0,0,0,0,0,0,0]
    agent5_actions = [0,0,0,0,0,0,0]
    agent6_actions = [0,0,0,0,0,0,0]
    agent7_actions = [0,0,0,0,0,0,0]
    agent8_actions = [0,0,0,0,0,0,0]
    agent9_actions = [0,0,0,0,0,0,0]

    decider_matrix = [0,0,0,0]

    # [judge majority in wood group as choppers, judge majority in wood group as miners, 
    # judge minority of the wood group as choppers, judge minority of the wood group as miners,
    # judge majority of the stone group as choppers, judge majority of the stone group as miners, 
    # judge minority of the stone group as choppers, judge minority of the stone group as miners]
    interaction_record = [0,0,
                        0,0,
                        0,0,
                        0,0] 


    decider_reward_record = [] 

    decider_decision = [] 
    record_agent1_actions = [] 
    record_agent2_actions = [] 
    record_agent3_actions = [] 
    record_agent4_actions = [] 
    record_agent5_actions = [] 
    record_agent6_actions = [] 
    record_agent7_actions = [] 
    record_agent8_actions = [] 
    record_agent9_actions = [] 

    # Dependent variables data
    policy_normalized_reward_record = [] 
    total_normalized_reward_record = [] 
    interaction_records = [] 

    # Training loop for each run
    for iteration in range(int(1/replace_prop)+1+1): 
        decider_reward_record = [] 

        decider_decision = [] 
        record_agent1_actions = [] 
        record_agent2_actions = [] 
        record_agent3_actions = [] 
        record_agent4_actions = [] 
        record_agent5_actions = [] 
        record_agent6_actions = [] 
        record_agent7_actions = [] 
        record_agent8_actions = [] 
        record_agent9_actions = [] 

        # dependent variables data 
        policy_normalized_reward_record = [] 
        total_normalized_reward_record = [] 
        interaction_records = [] 
        if iteration == int(1/replace_prop)+1: 
            decider_model.save(model_path+f'{run}_decider_model.pkl')
            decider_model = PPO(
                    device=device, 
                    state_dim=16,
                    action_dim=2,
                    lr_actor=0.001,
                    lr_critic=0.0005,
                    gamma=0.9,
                    K_epochs=10,
                    eps_clip=0.2 
                )
            
            
            decider_model.replay = RolloutBuffer() 

            decider_model.model1.to(device)
            
        for epoch in range(num_epoch):
            agent_action_epoch = [[0,0,0,0,0,0,0] for i in range(population_size)]
            agent_reward_epoch = [0 for i in range(population_size)]

            done = 0

            env.wood = 10
            env.stone = 10

            # Initialize the number of wood, stones, and coins
            for agent in range(len(agent_list)):
                agent_list[agent].coin = 0
                agent_list[agent].wood = 0
                agent_list[agent].stone = 0
                if agent_list[agent].policy in [2,5,8]:
                    agent_list[agent].coin = 6
                

            turn = 0
            while done != 1:
                turn = turn + 1
                if turn > max_turns:
                    done = 1
                
                action_order = [i for i in range(len(agent_list))] 
                random.shuffle(action_order)
                for agent in action_order:

                    cur_wood = agent_list[agent].wood
                    cur_stone = agent_list[agent].stone
                    cur_coin = agent_list[agent].coin

                    state, previous_state = generate_input(agent_list, agent, agent_list[agent].state)
                    state = state.unsqueeze(0).to(device)
                    action, action_logprob = models[agent].take_action(state)
                
                    pred_success = True
                    if action in (3,4):
                        decider_state = torch.tensor(agent_list[agent].appearance).float().to(device)
                        decider_action, decider_action_logprob = decider_model.take_action(decider_state)
                        
                        agent_action = action - 3
                        decider_reward = 1

                        not_suf = True 
                        if (action == 3 and agent_list[agent].wood > 1) or (action == 4 and agent_list[agent].stone > 1):
                            not_suf = False 

                        if decider_action != agent_action:
                            decider_reward = -1
                            pred_success = False 
                        if decider_action == agent_action and not_suf:
                            decider_reward = -.3

                        if decider_action == 0 and agent_action == 0:
                            decider_matrix[0] = decider_matrix[0] + 1
                        if decider_action == 1 and agent_action == 0:
                            decider_matrix[1] = decider_matrix[1] + 1
                        if decider_action == 0 and agent_action == 1:
                            decider_matrix[2] = decider_matrix[2] + 1
                        if decider_action == 1 and agent_action == 1:
                            decider_matrix[3] = decider_matrix[3] + 1
                        
                        if decider_action == 0:
                            if agent_list[agent].policy == 0: 
                                interaction_record[0] += 1 
                            elif agent_list[agent].policy == 1: 
                                interaction_record[2] += 1 
                            elif agent_list[agent].policy == 3: 
                                interaction_record[6] += 1 
                            elif agent_list[agent].policy == 4: 
                                interaction_record[4] += 1 
                        elif decider_action == 1: 
                            if agent_list[agent].policy == 0: 
                                interaction_record[1] += 1
                            elif agent_list[agent].policy == 1: 
                                interaction_record[3] += 1 
                            elif agent_list[agent].policy == 3: 
                                interaction_record[7] += 1 
                            elif agent_list[agent].policy == 4: 
                                interaction_record[5] += 1 


                        decider_model.replay.states.append(decider_state)
                        decider_model.replay.actions.append(decider_action)
                        decider_model.replay.logprobs.append(decider_action_logprob)
                        decider_model.replay.rewards.append(decider_reward)
                        decider_model.replay.is_terminals.append(done)           
                        decider_rewards += decider_reward  
                        
                    env, reward, next_state, done, new_loc = agent_list[agent].transition(env, models, action, done, [], agent_list, agent, pred_success)
                    agent_list[agent].episode_memory.states.append(state)
                    agent_list[agent].episode_memory.actions.append(action)
                    agent_list[agent].episode_memory.logprobs.append(action_logprob)
                    agent_list[agent].episode_memory.rewards.append(reward)
                    agent_list[agent].episode_memory.is_terminals.append(done)
                    rewards[agent_list[agent].policy] = rewards[agent_list[agent].policy] + reward
                    

                    # should make this a matrix to clean up code
                    if agent_list[agent].policy == 0:
                        agent1_actions[action] = agent1_actions[action] + 1
                    if agent_list[agent].policy == 1:
                        agent2_actions[action] = agent2_actions[action] + 1
                    if agent_list[agent].policy == 2:
                        agent3_actions[action] = agent3_actions[action] + 1

                    if agent_list[agent].policy == 3:
                        agent4_actions[action] = agent4_actions[action] + 1
                    if agent_list[agent].policy == 4:
                        agent5_actions[action] = agent5_actions[action] + 1
                    if agent_list[agent].policy == 5:
                        agent6_actions[action] = agent6_actions[action] + 1

                    if agent_list[agent].policy == 6:
                        agent7_actions[action] = agent7_actions[action] + 1
                    if agent_list[agent].policy == 7:
                        agent8_actions[action] = agent8_actions[action] + 1
                    if agent_list[agent].policy == 8:
                        agent9_actions[action] = agent9_actions[action] + 1

                    # record individual agent's actions
                    agent_action_epoch[agent][action] += 1
                    agent_reward_epoch[agent] += reward 

            # Update the agent and the decider's models
            for count, model in enumerate(models):
                loss = model.training(agent_list[count].episode_memory, entropy_coefficient=0.01)
                agent_list[count].episode_memory.clear() 
                losses = losses + loss.detach().cpu().numpy()

            decider_loss = decider_model.training(decider_model.replay, entropy_coefficient=0.01)        
            decider_losses = decider_losses + decider_loss.detach().cpu().numpy() 
            decider_model.replay.clear() 

            agent_action_list_to_concat = pd.Series([run, iteration, epoch]+agent_action_epoch, index=['run', 'iteration', 'epoch']+[f"Agent{i+1+min(int(iteration*population_size*replace_prop), population_size)}"for i in range(population_size)])
            agent_actions.loc[len(agent_actions)] = agent_action_list_to_concat
            agent_reward_list_to_concat = pd.Series([run, iteration, epoch]+agent_reward_epoch, index=['run', 'iteration', 'epoch']+[f"Agent{i+1+min(int(iteration*population_size*replace_prop), population_size)}"for i in range(population_size)])
            agent_rewards.loc[len(agent_rewards)] = agent_reward_list_to_concat
            
            # Save all the files
            agent_actions.to_csv(data_path+'all_agent_actions.csv', index=False, sep=',')
            agent_rewards.to_csv(data_path+'all_agent_rewards.csv', index=False, sep=',')

         

            if (epoch % unit_test_frequency == 0) or (iteration == 6 and epoch == num_epoch-1): 
                unit_test_record = pd.DataFrame(columns=['test_index', 'id', 'group', 'subgroup', 'skill', 'unseen', 'guess', 'guess_wood_prob', 'guess_stone_prob'])

                guess, guess_prob = unit_test(decider_model=decider_model, agent_list=agent_list, unseen_agents=unseen_agents)
                guess = [int(val) for val in list(guess)] 
                guess_prob = [float(val) for val in list(guess_prob)]
                
                # Save data
                test_n = epoch//unit_test_frequency + 1 + iteration * (num_epoch/unit_test_frequency)
                unseen = [0 for i in range(population_size)] + [1 for i in range(n_unseen_agents)]
                skill = [agent.wood_skill for agent in agent_list] + [-1 for i in range(n_unseen_agents)]
                group = [agent.agent_type//3 for agent in agent_list] + unseen_agent_groups 
                subgroup = [agent.agent_type for agent in agent_list] + [-1 for i in range(n_unseen_agents)] 
                id = [agent.appearance for agent in agent_list] + unseen_agents 
                for val in id:
                    if id.count(val) > 1: 
                        print('element duplicate:', id.count(val))
                for n in range(population_size+n_unseen_agents): 
                    if guess[n] == 0: 
                        guess_wood_prop = guess_prob[n] 
                    else: 
                        guess_wood_prop = 1-guess_prob[n]
                    info_to_concat = [test_n, id[n], group[n], subgroup[n], skill[n], unseen[n], guess[n], guess_wood_prop, 1-guess_wood_prop]
                    unit_test_record.loc[len(unit_test_record)] = info_to_concat 
                unit_test_record.to_csv(unit_test_path+f'{run}_{test_n}_unit_test.csv', index=False, sep=',')

            # Print training progress
            if epoch % 1 == 0:
                print("--------------------------------------")
                r1 = [round(a/b, 2) for a,b in zip(rewards,num_agent_per_policy)]
                r2 = sum([rewards[i] for i in [0,1,3,4]])/number_agent_of_interest
                print("condition:", population_size, prop)
                print("number of agents in each subgroup:", num_agent_per_policy)
                print("run:", run, 'iteration:', iteration, "epoch:" , epoch, "loss: ",losses, "decider loss: ", decider_losses, "\n", "points (wood, stone, house): ", [round(a/b, 2) for a,b in zip(rewards,num_agent_per_policy)])
                print('normalized benefits of the microsociety:', r2)
                print("chop, mine, build, sell_wood, sell_stone, buy_wood, buy_stone")
                print("agent1 behaviours - chop_c: ", agent1_actions)
                print("agent2 behaviours - chop_m: ", agent2_actions)
                print("agent3 behaviours - chop_h: ", agent3_actions)
                print("agent4 behaviours - mine_c: ", agent4_actions)
                print("agent5 behaviours - mine_m: ", agent5_actions)
                print("agent6 behaviours - mine_h: ", agent6_actions)
                print("agent7 behaviours - hous_c: ", agent7_actions)
                print("agent8 behaviours - hous_m: ", agent8_actions)
                print("agent9 behaviours - hous_h: ", agent9_actions)
                print("decider maxtrx: ", decider_matrix)
                print("Guess wood -- majority of wood group:", interaction_record[0], "  Guess stone -- majority of wood group:",
                    interaction_record[1], ' ', interaction_record[0]/(interaction_record[0]+interaction_record[1]+1e-7))
                print("Guess wood -- minority of wood group:", interaction_record[2], "  Guess stone -- minority of wood group:",
                    interaction_record[3], ' ', interaction_record[2]/(interaction_record[2]+interaction_record[3]+1e-7))
                print("Guess wood -- majority of stone group:", interaction_record[4], "  Guess stone -- majority of stone group:", interaction_record[5], ' ', interaction_record[5]/(interaction_record[4]+interaction_record[5]+1e-7))
                print("Guess wood -- minority of stone group:", interaction_record[6], "  Guess stone -- minority of stone group:", interaction_record[7], ' ', interaction_record[7]/(interaction_record[6]+interaction_record[7]+1e-7))
                
                record_agent1_actions.append(agent1_actions)
                record_agent2_actions.append(agent2_actions)
                record_agent3_actions.append(agent3_actions)
                record_agent4_actions.append(agent4_actions)
                record_agent5_actions.append(agent5_actions)
                record_agent6_actions.append(agent6_actions)
                record_agent7_actions.append(agent7_actions)
                record_agent8_actions.append(agent8_actions)
                record_agent9_actions.append(agent9_actions)
                policy_normalized_reward_record.append(r1)
                total_normalized_reward_record.append(r2)
                decider_reward_record.append(decider_rewards)
                decider_decision.append(decider_matrix) 
                interaction_records.append(interaction_record)

                rewards = [0,0,0,0,0,0,0,0,0]
                decider_rewards = 0 
                losses = 0
                decider_losses = 0
                agent1_actions = [0,0,0,0,0,0,0]
                agent2_actions = [0,0,0,0,0,0,0]
                agent3_actions = [0,0,0,0,0,0,0]
                agent4_actions = [0,0,0,0,0,0,0]
                agent5_actions = [0,0,0,0,0,0,0]
                agent6_actions = [0,0,0,0,0,0,0]
                agent7_actions = [0,0,0,0,0,0,0]
                agent8_actions = [0,0,0,0,0,0,0]
                agent9_actions = [0,0,0,0,0,0,0]
                decider_matrix = [0,0,0,0]
                interaction_record = [0,0,0,0,0,0,0,0]


     

        # Replace 10% of the old agents with new ones
        if iteration < int(1/replace_prop):
            for j in range(int(population_size*replace_prop)): 
                model_name = model_path+f"{run}_Agent{int(iteration*population_size*replace_prop+j+1)}_model.pkl"
                models[j].save(model_name)
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
        agent_property = pd.DataFrame(columns=['agent_type', 'number_iteration'])
        for j, agent in enumerate(agent_list): 
            agent_property.loc[len(agent_property)] = [agent_list[j].agent_type, iteration]
        agent_property.to_csv(data_path+f'{run}_{iteration}_agent_info.csv', index=False, sep=',')
        
        if iteration < int(1/replace_prop):
            replace_agents_equal_ratio(agent_list=agent_list, population_size=population_size, prop=0.45, prop_reverse=0.45,
                                    replace_prop=replace_prop, attributes=replacement_attributes, iteration=iteration)
         
        for i in range(9): 
            num_agent_per_policy[i] = len([agent for agent in agent_list if agent.policy==i]) 
        print(num_agent_per_policy)   

        # Save group-level data
        save_dir = dir_group_data
        np.savetxt(save_dir+f"{population_size}_{prop}_{run}_{iteration}_agent1_actions.csv", np.array(record_agent1_actions), delimiter=',', fmt='%s')
        np.savetxt(save_dir+f"{population_size}_{prop}_{run}_{iteration}_agent2_actions.csv", np.array(record_agent2_actions), delimiter=',', fmt='%s')
        np.savetxt(save_dir+f"{population_size}_{prop}_{run}_{iteration}_agent3_actions.csv", np.array(record_agent3_actions), delimiter=',', fmt='%s')
        np.savetxt(save_dir+f"{population_size}_{prop}_{run}_{iteration}_agent4_actions.csv", np.array(record_agent4_actions), delimiter=',', fmt='%s')
        np.savetxt(save_dir+f"{population_size}_{prop}_{run}_{iteration}_agent5_actions.csv", np.array(record_agent5_actions), delimiter=',', fmt='%s')
        np.savetxt(save_dir+f"{population_size}_{prop}_{run}_{iteration}_agent6_actions.csv", np.array(record_agent6_actions), delimiter=',', fmt='%s')
        np.savetxt(save_dir+f"{population_size}_{prop}_{run}_{iteration}_agent7_actions.csv", np.array(record_agent7_actions), delimiter=',', fmt='%s')
        np.savetxt(save_dir+f"{population_size}_{prop}_{run}_{iteration}_agent8_actions.csv", np.array(record_agent8_actions), delimiter=',', fmt='%s')
        np.savetxt(save_dir+f"{population_size}_{prop}_{run}_{iteration}_agent9_actions.csv", np.array(record_agent9_actions), delimiter=',', fmt='%s')
        np.savetxt(save_dir+f"{population_size}_{prop}_{run}_{iteration}_decider_matrix.csv", np.array(decider_decision), delimiter=',', fmt='%s')

        np.savetxt(save_dir+f"{population_size}_{prop}_{run}_{iteration}_total_normalized_agent_rewards.csv", np.array(total_normalized_reward_record),  delimiter=',', fmt='%s') 
        np.savetxt(save_dir+f"{population_size}_{prop}_{run}_{iteration}_policy_normalized_agent_rewards.csv", np.array(policy_normalized_reward_record), delimiter=',', fmt='%s')
        np.savetxt(save_dir+f"{population_size}_{prop}_{run}_{iteration}_decider_rewards.csv", np.array(decider_reward_record), delimiter=',', fmt='%s')
        np.savetxt(save_dir+f"{population_size}_{prop}_{run}_{iteration}_interaction_record.csv", np.array(interaction_records), delimiter=',', fmt='%s')


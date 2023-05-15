# import modules
import random
import numpy as np
import torch
import pandas as pd
from collections import deque

from examples.ai_economist_simple.elements_fixed import Agent
from examples.ai_economist_simple.env import AIEcon_simple_game
from examples.ai_economist_simple.env import generate_input
from examples.ai_economist_simple.PPO import RolloutBuffer, PPO


# set directory
data_path_ind = 'save/'
data_path_group = 'save/'
individuation_data = 'save/'
individuation_model = 'save/'

# training and environment parameters
device = "cpu"
print(device)
num_epoch = 200
num_run = 30
population_size_values = [30, 100]
prop = 0.5
prop_reverse = 0.75-prop
n_unseen_agents = 120

    
# main loop
type_record = [] 
for run in range(num_run):  
    for population_size in population_size_values:
    
        if population_size in (300, 600):
            n_agents_per_group = population_size/3
            wood_group_distribution = [n_agents_per_group*prop, n_agents_per_group*prop_reverse, n_agents_per_group*(1-prop-prop_reverse)]
            stone_group_distribution = [n_agents_per_group*prop_reverse, n_agents_per_group*prop, n_agents_per_group*(1-prop-prop_reverse)]
            house_group_distribution = [0.1, 0.1, 0.8]
            agent_distribution = {'wood': wood_group_distribution, 'stone': stone_group_distribution, 'house': house_group_distribution}
        elif population_size == 30:
            agent_distribution = {'wood': [6, 3, 1], 'stone': [3, 6, 1], 'house': [1, 1, 8]}
        elif population_size == 100:
            agent_distribution = {'wood': [16, 8, 8], 'stone': [8, 16, 8], 'house':[4, 4, 28]}
 
        if run < - 1:
    	    agent_actions = pd.read_csv(data_path_ind+'all_agent_actions.csv')
    	    agent_rewards = pd.read_csv(data_path_ind+'all_agent_rewards.csv')
    	    individuation_record = pd.read_csv(individuation_data+'main_data.csv')
        else:
        	agent_actions = pd.DataFrame(columns=['run', 'epoch']+[f"Agent{i+1}"for i in range(population_size)])
        	agent_rewards = pd.DataFrame(columns=['run', 'epoch']+[f"Agent{i+1}"for i in range(population_size)])
        	individuation_record = pd.DataFrame(columns=['id', 'subtype', 'epoch', 'run', 'action', 'guess'])
        
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

        models = create_models(population_size)

        env = AIEcon_simple_game()

        i11, i12, i13 = 0, 0, 0


        binary = [0,1]
        
        # create the attributes for all the agents that will occur in the environment 
        appearence_all = [] 
        num_agent_per_policy = [0 for i in range(9)]
        all_individual_attributes = [[i1, i2, i3, i4, i5, i6, i7, i8, i9, i10]
                                    for i1 in binary for i2 in binary for i3 in binary 
                                    for i4 in binary for i5 in binary for i6 in binary 
                                    for i7 in binary for i8 in binary for i9 in binary 
                                    for i10 in binary]

        random.shuffle(all_individual_attributes)

        original_agents_attributes, unseen_attributes = all_individual_attributes[:population_size], all_individual_attributes[population_size:population_size*2]
        original_agents_attributes = [list(tuple(i)+(i11, i12, i13)) for i in original_agents_attributes]
        unseen_attributes = [list(tuple(i)+(i11, i12, i13)) for i in unseen_attributes]

        # create the initial agents 
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

        max_turns = 50

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

        # training loop for each epoch
        for epoch in range(num_epoch):
            agent_action_epoch = [[0,0,0,0,0,0,0] for i in range(population_size)]
            agent_reward_epoch = [0 for i in range(population_size)]

            done = 0

            env.wood = 10
            env.stone = 10

            # initialize the number of wood, stones, and coins 
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

                        # save guesses and sells 
                        id = agent_list[agent].appearance 
                        subtype = agent_list[agent].policy
                        agent_interaction_list_to_concat = [id, subtype, epoch, run, int(agent_action), int(decider_action)] 
                        individuation_record.loc[len(individuation_record)] = agent_interaction_list_to_concat

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

            # update the agent and the decider's models 
            for count, model in enumerate(models):
                loss = model.training(agent_list[count].episode_memory, entropy_coefficient=0.01)
                agent_list[count].episode_memory.clear() 
                losses = losses + loss.detach().cpu().numpy()

            decider_loss = decider_model.training(decider_model.replay, entropy_coefficient=0.01)        
            decider_losses = decider_losses + decider_loss.detach().cpu().numpy() 
            decider_model.replay.clear() 

            
            agent_action_list_to_concat = pd.Series([run, epoch]+agent_action_epoch, index=['run', 'epoch']+[f'Agent{i+1}'for i in range(population_size)])
            agent_actions.loc[len(agent_actions)] = agent_action_list_to_concat
            agent_reward_list_to_concat = pd.Series([run, epoch]+agent_reward_epoch, index=['run', 'epoch']+[f'Agent{i+1}'for i in range(population_size)])
            agent_rewards.loc[len(agent_rewards)] = agent_reward_list_to_concat
    
            # save all the files
            if epoch == num_epoch - 1:
                agent_actions.to_csv(data_path_ind+f'{run}_{population_size}_all_agent_actions.csv', index=False, sep=',')
                agent_rewards.to_csv(data_path_ind+f'{run}_{population_size}_all_agent_rewards.csv', index=False, sep=',')
                individuation_record.to_csv(individuation_data+f'{run}_{population_size}_main_data.csv', index=False, sep=',')
        

            # print training progress
            if epoch % 1 == 0:
                print("--------------------------------------")
                r1 = [round(a/b, 2) for a,b in zip(rewards,num_agent_per_policy)]
                r2 = sum([rewards[i] for i in [0,1,3,4]])/number_agent_of_interest
                print("condition:", population_size, prop)
                print("number of agents in each subgroup:", num_agent_per_policy)
                print("run:", run, "epoch:" , epoch, "loss: ",losses, "decider loss: ", decider_losses, "\n", "points (wood, stone, house): ", [round(a/b, 2) for a,b in zip(rewards,num_agent_per_policy)])
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
                print("Guess wood -- majority of wood group:", interaction_record[0], "  Guess stone -- majority of wood group:", interaction_record[1], ' ', interaction_record[0]/(interaction_record[0]+interaction_record[1]+1e-7))
                print("Guess wood -- minority of wood group:", interaction_record[2], "  Guess stone -- minority of wood group:", interaction_record[3], ' ', interaction_record[2]/(interaction_record[2]+interaction_record[3]+1e-7))
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

        for i in range(9): 
            num_agent_per_policy[i] = len([agent for agent in agent_list if agent.policy==i]) 
        print(num_agent_per_policy)   

        agent_property = pd.DataFrame(columns=['agent_type', 'id'])
        for j, agent in enumerate(agent_list): 
            agent_property.loc[len(agent_property)] = [agent_list[j].agent_type, agent_list[j].appearance]
        agent_property.to_csv(individuation_data+f'{run}_agent_info.csv', index=False, sep=',')

        model_name = individuation_model+f"{run}_{population_size}_decider_model.pkl"
        decider_model.save(model_name)


        save_dir = data_path_group
        np.savetxt(save_dir+f"{population_size}_{prop}_{run}_agent1_actions.csv", np.array(record_agent1_actions), delimiter=',', fmt='%s')
        np.savetxt(save_dir+f"{population_size}_{prop}_{run}_agent2_actions.csv", np.array(record_agent2_actions), delimiter=',', fmt='%s')
        np.savetxt(save_dir+f"{population_size}_{prop}_{run}_agent3_actions.csv", np.array(record_agent3_actions), delimiter=',', fmt='%s')
        np.savetxt(save_dir+f"{population_size}_{prop}_{run}_agent4_actions.csv", np.array(record_agent4_actions), delimiter=',', fmt='%s')
        np.savetxt(save_dir+f"{population_size}_{prop}_{run}_agent5_actions.csv", np.array(record_agent5_actions), delimiter=',', fmt='%s')
        np.savetxt(save_dir+f"{population_size}_{prop}_{run}_agent6_actions.csv", np.array(record_agent6_actions), delimiter=',', fmt='%s')
        np.savetxt(save_dir+f"{population_size}_{prop}_{run}_agent7_actions.csv", np.array(record_agent7_actions), delimiter=',', fmt='%s')
        np.savetxt(save_dir+f"{population_size}_{prop}_{run}_agent8_actions.csv", np.array(record_agent8_actions), delimiter=',', fmt='%s')
        np.savetxt(save_dir+f"{population_size}_{prop}_{run}_agent9_actions.csv", np.array(record_agent9_actions), delimiter=',', fmt='%s')
        np.savetxt(save_dir+f"{population_size}_{prop}_{run}_decider_matrix.csv", np.array(decider_decision), delimiter=',', fmt='%s')

        np.savetxt(save_dir+f"{population_size}_{prop}_{run}_total_normalized_agent_rewards.csv", np.array(total_normalized_reward_record),  delimiter=',', fmt='%s') 
        np.savetxt(save_dir+f"{population_size}_{prop}_{run}_policy_normalized_agent_rewards.csv", np.array(policy_normalized_reward_record), delimiter=',', fmt='%s')
        np.savetxt(save_dir+f"{population_size}_{prop}_{run}_decider_rewards.csv", np.array(decider_reward_record), delimiter=',', fmt='%s')
        np.savetxt(save_dir+f"{population_size}_{prop}_{run}_interaction_record.csv", np.array(interaction_records), delimiter=',', fmt='%s')

        


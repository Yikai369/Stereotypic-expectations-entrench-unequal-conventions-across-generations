"""Study 1."""
import random
import torch

from env import AIEcon_simple_game, generate_input
from PPO import RolloutBuffer, PPO
import utils 

import study1_config 


def study1(
        dir_group_data='save/'
        ):
    # set directory
    path_group_data = dir_group_data

    # training and environment parameters
    device = "cpu"
    print(device)

    study_number = study1_config.study_number 
    num_epoch = study1_config.num_epoch
    num_run = study1_config.num_run
    max_turns = study1_config.max_turns
    population_sizes = study1_config.population_sizes
    prop = study1_config.prop
    prop_reverse = study1_config.prop_reverse
    num_subgroups = study1_config.num_subgroups 
    n_unseen_agents = study1_config.n_unseen_agents 

    # Main loop
    for run in range(num_run):
        for population_size in population_sizes:
            
            # Generate the specific agent distribution for each group 
            agent_distribution = utils.generate_agent_distribution(
                prop, 
                prop_reverse, 
                population_size
                )

            # Create a model for the market decider  
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
            decider_model.model1.to(device)

            # Create the memory buffer for the market 
            decider_model.replay = RolloutBuffer()

            # Create models for the agents 
            models = utils.create_models(population_size, device)

            # Initialize the game environment 
            env = AIEcon_simple_game()
            
            # Create the attributes (identity code) for all the agents that will occur in the environment
            original_agents_attributes, _, _, _ = utils.create_identity_feature(
                study_number,
                13, 
                population_size,
                n_unseen_agents
                )

            # Create agents and the corresponding models
            agent_list = utils.create_agents(
                population_size, 
                agent_distribution, 
                original_agents_attributes, 
                study_number
                )
            num_agents = len(agent_list)
            print('Total number of agents:', num_agents)

            # Check the agent population 
            num_agent_per_policy = [0 for _ in range(num_subgroups)]
            for i in range(num_subgroups):
                num_agent_per_policy[i] = len([agent for agent in agent_list if agent.policy==i])
            number_agent_of_interest = 0
            for i in range(num_subgroups):
                if i in [0,1,3,4]:
                    number_agent_of_interest += num_agent_per_policy[i]
            print('Number of agents to examine:', number_agent_of_interest)
            print('Number of agents in each subgroup:', num_agent_per_policy)

            # Variables to record history in each epoch 
            rewards = [0,0,0,0,0,0,0,0,0]
            decider_rewards = 0
            losses = 0
            decider_losses = 0
            all_agent_actions = [[i for i in range(env.action_space)] 
                                for _ in range(num_subgroups)] 
            interaction_record = [0,0,0,0,0,0,0,0] 

            # Variables to record history of all epochs in each condition 
            decider_reward_sum = []
            all_agent_actions_sum = [[] for _ in range(num_subgroups)]
            policy_normalized_reward_sum = []
            total_normalized_reward_sum = []
            interaction_record_sum = []

            # Training loop for each run
            for epoch in range(num_epoch):

                done = 0

                env.wood = 10
                env.stone = 10

                # Set the intial values of agents' possessions 
                for agent in range(len(agent_list)):
                    agent_list[agent].coin = 0
                    agent_list[agent].wood = 0
                    agent_list[agent].stone = 0
                    if agent_list[agent].policy in [2,5,8]:
                        agent_list[agent].coin = 6
                    
                # Loop for each epoch 
                turn = 0
                while done != 1:
                    turn = turn + 1
                    if turn > max_turns:
                        done = 1
                    
                    # In each step, every agent takes an action 
                    action_order = [i for i in range(len(agent_list))]
                    random.shuffle(action_order)
                    for agent in action_order:

                        state, _ = generate_input(
                            agent_list, agent, 
                            agent_list[agent].state
                            )
                        state = state.unsqueeze(0).to(device)
                        action, action_logprob = models[agent].take_action(state)

                        # The market decider makes a prediction when an agent chooses selling
                        pred_success = True
                        if action in (3,4):
                            decider_state = torch.tensor(agent_list[agent].appearance).float().to(device)
                            decider_action, decider_action_logprob = decider_model.take_action(decider_state)
                            
                            agent_action = action - 3
                            decider_reward = 1

                            # Check whether the agent has suffcient resources to make a sale
                            not_suf = True
                            if (action == 3 and agent_list[agent].wood > 1) \
                                or (action == 4 and agent_list[agent].stone > 1):
                                not_suf = False

                            # The market decider get punished if the prediction is wrong
                            if decider_action != agent_action:
                                decider_reward = -1
                                pred_success = False
                            # A slight punishment to the market if the agent lacks resources
                            if decider_action == agent_action and not_suf:
                                decider_reward = -.3

                            # Record the history of the interactions between the agents and the market
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

                            # Update the market's memory buffer
                            decider_model.replay.states.append(decider_state)
                            decider_model.replay.actions.append(decider_action)
                            decider_model.replay.logprobs.append(decider_action_logprob)
                            decider_model.replay.rewards.append(decider_reward)
                            decider_model.replay.is_terminals.append(done)
                            decider_rewards += decider_reward
                        
                        # Transit to next state based on the selected actions of the agent and the market 
                        env, reward, _, done = agent_list[agent].transition(
                            env, 
                            models, 
                            action, 
                            done, 
                            agent_list, 
                            agent, 
                            pred_success
                            )

                        # Update the agent's memory buffer
                        agent_list[agent].episode_memory.states.append(state)
                        agent_list[agent].episode_memory.actions.append(action)
                        agent_list[agent].episode_memory.logprobs.append(action_logprob)
                        agent_list[agent].episode_memory.rewards.append(reward)
                        agent_list[agent].episode_memory.is_terminals.append(done)
                        
                        # Record agent actions and rewards 
                        all_agent_actions[agent_list[agent].policy][action] += 1 
                        rewards[agent_list[agent].policy] += reward

                # Update the agents' models
                for count, model in enumerate(models):
                    loss = model.training(
                        agent_list[count].episode_memory, 
                        entropy_coefficient=0.01
                        )
                    agent_list[count].episode_memory.clear()
                    losses = losses + loss.detach().cpu().numpy()

                # Update the market decider's model
                decider_loss = decider_model.training(
                    decider_model.replay, 
                    entropy_coefficient=0.01
                    )
                decider_losses = decider_losses + decider_loss.detach().cpu().numpy()
                decider_model.replay.clear()

                # Print training progress
                if study_number != 3: 
                    iteration = 0 
                subgroup_r, total_r = utils.show_training_progress(
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
                                                            ) 
                
                # Update the history of this condition  
                history = [ rewards, 
                            decider_rewards, 
                            losses, 
                            decider_losses, 
                            interaction_record, 
                            all_agent_actions
                            ]
                utils.update_history(
                            subgroup_r, 
                            total_r,
                            all_agent_actions_sum, 
                            policy_normalized_reward_sum,
                            total_normalized_reward_sum, 
                            decider_reward_sum, 
                            interaction_record_sum, 
                            history
                            )
                
                # Clean the history of this epoch 
                rewards = [0,0,0,0,0,0,0,0,0]
                decider_rewards = 0
                losses = 0
                decider_losses = 0
                all_agent_actions = [[i for i in range(env.action_space)] 
                                    for _ in range(num_subgroups)] 
                interaction_record = [0,0,0,0,0,0,0,0] 
                
            # Save 
            utils.save_data(
                path_group_data, 
                study_number,
                population_size, 
                prop, 
                run, 
                all_agent_actions_sum, 
                total_normalized_reward_sum, 
                policy_normalized_reward_sum, 
                decider_reward_sum, 
                interaction_record_sum)


if __name__ == '__main__':
    study1() 

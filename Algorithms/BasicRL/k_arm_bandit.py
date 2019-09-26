##############################################################
# This module contains all functions for solving
# K arm bandit problems as described in Richard Suttons Book
##############################################################

import numpy as np

from scipy.stats import norm

def run_e_greedy_bandit(lever_data,epochs,epsilon,init_q_value,algo_type):
    '''
    This is a function for running the epsilon greedy bandit with different
    parameters.

    :param lever_data: Dictionary of the parameters used for each lever i.e (mean , std )
    :param epochs: number of iterations
    :param epsilon: for e-greedy function
    :param init_q_value: Initial guess for the value of each lever
    :param algo_type: UCB or greedy
    :return:
    '''

    lever_data = lever_data['data']

    arr = np.array([0.0])  # init avg_reward

    count_values = np.zeros(len(lever_data))

    q_values = np.ones(len(lever_data)) * init_q_value

    avg_reward = 0
    for i in range(1, epochs + 1):
        print("Epoch: " + str(i))

        print("Avg Reward: " + str(avg_reward))

        print("Reward Vector: " + str(q_values))

        rand_uniform = np.random.uniform(0, 1)

        action = select_action(q_values, count_values, epsilon, len(lever_data), i, algo_type, 1)

        mean_current_lever = lever_data[action]['mean']

        std_current_lever = lever_data[action]['std']

        reward = norm.rvs(loc=mean_current_lever, scale=std_current_lever)

        # update lever average and count
        count_values[action] += 1

        q_values[action] = q_values[action] + (1 / count_values[action]) * (reward - q_values[action])

        # Update overall average reward
        avg_reward = avg_reward + (1.0 / i) * (reward - avg_reward)


    return data, avg_reward, list(q_values), list(count_values)


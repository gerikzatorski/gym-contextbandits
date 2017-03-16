#!/usr/bin/env python
# basic_rl.py (v0.0.1)

import argparse
import gym_contextbandits
import gym_bandits

parser = argparse.ArgumentParser(description='Use SARSA/Q-learning algorithm with epsilon-greedy/softmax polciy.')
parser.add_argument('-a', '--algorithm', default='sarsa', choices=['sarsa', 'q_learning'],
                    help="Type of learning algorithm. (Default: sarsa)")
parser.add_argument('-p', '--policy', default='epsilon_greedy', choices=['epsilon_greedy', 'softmax'],
                    help="Type of policy. (Default: epsilon_greedy)")
parser.add_argument('-e', '--environment', default='ContextBandits-v0',
                    help="Name of the environment provided in the OpenAI Gym. (Default: FrozenLake-v0)")
parser.add_argument('-n', '--nepisode', default='5000', type=int,
                    help="Number of episode. (Default: 5000)")
parser.add_argument('-al', '--alpha', default='0.1', type=float,
                    help="Learning rate. (Default: 0.1)")
parser.add_argument('-be', '--beta', default='0.0', type=float,
                    help="Initial value of an inverse temperature. (Default: 0.0)")
parser.add_argument('-bi', '--betainc', default='0.02', type=float,
                    help="Linear increase rate of an inverse temperature. (Default: 0.02)")
parser.add_argument('-ga', '--gamma', default='0.99', type=float,
                    help="Discount rate. (Default: 0.99)")
parser.add_argument('-ep', '--epsilon', default='0.5', type=float,
                    help="Fraction of random exploration in the epsilon greedy. (Default: 0.5)")
parser.add_argument('-ed', '--epsilondecay', default='0.999', type=float,
                    help="Decay rate of epsilon in the epsilon greedy. (Default: 0.999)")
parser.add_argument('-ms', '--maxstep', default='100', type=int,
                    help="Maximum step allowed in a episode. (Default: 100)")
args = parser.parse_args()

import gym
import numpy as np
import os

import matplotlib.pyplot as plt

def softmax(q_value, beta=1.0):
    assert beta >= 0.0
    q_tilde = q_value - np.max(q_value)
    factors = np.exp(beta * q_tilde)
    return factors / np.sum(factors)

def select_a_with_softmax(curr_s, q_value, beta=1.0):
    prob_a = softmax(q_value[curr_s, :], beta=beta)
    cumsum_a = np.cumsum(prob_a)
    return np.where(np.random.rand() < cumsum_a)[0][0]

def select_a_with_epsilon_greedy(curr_s, q_value, epsilon=0.1):
    a = np.argmax(q_value[curr_s, :])
    if np.random.rand() < epsilon:
        a = np.random.randint(q_value.shape[1])
    return a

def main():

    env_type = args.environment
    algorithm_type = args.algorithm
    policy_type = args.policy

    
    # Random seed
    np.random.RandomState(42)

    # Selection of the problem
    env = gym.envs.make(env_type)

    
    
    # Constraints imposed by the environment
    n_a = env.action_space.n
    n_s = env.observation_space.n

    # Meta parameters for the RL agent
    alpha = args.alpha
    beta = init_beta = args.beta
    beta_inc = args.betainc
    gamma = args.gamma
    epsilon = args.epsilon
    epsilon_decay = args.epsilondecay

    # Experimental setup
    n_episode = args.nepisode
    # n_episode = 20
    print "n_episode ", n_episode
    max_step = args.maxstep

    # Initialization of a Q-value table
    q_value = np.zeros([n_s, n_a])

    # Initialization of a list for storing simulation history
    history = []

    print "algorithm_type: {}".format(algorithm_type)
    print "policy_type: {}".format(policy_type)

    env.reset()

    np.set_printoptions(precision=3, suppress=True)

    result_dir = 'results-{0}-{1}-{2}'.format(env_type, algorithm_type, policy_type)

    # Start monitoring the simulation for OpenAI Gym
    # env.monitor.start(result_dir, force=True)

    for i_episode in xrange(n_episode):

        # Reset a cumulative reward for this episode
        cumu_r = 0

        # Start a new episode and sample the initial state
        curr_s = env.reset()

        # Select the first action in this episode
        if policy_type == 'softmax':
            curr_a = select_a_with_softmax(curr_s, q_value, beta=beta)
        elif policy_type == 'epsilon_greedy':
            curr_a = select_a_with_epsilon_greedy(curr_s, q_value, epsilon=epsilon)
        else:
            raise ValueError("Invalid policy_type: {}".format(policy_type))

        for i_step in xrange(max_step):

            # Get a result of your action from the environment
            next_s, r, done, info = env.step(curr_a)

            print "{0} \t {1} \t {2} \t {3} \t {4}".format(curr_s, curr_a, r, done, info)


            # Modification of reward (not sure if it's OK to change reward setting by hand...)
            if done & (r == 0):
                # Punishment for falling into a hall
                r = 0.0
            elif not done:
                # Cost per step
                r = -0.001

            # Update a cummulative reward
            cumu_r = r + gamma * cumu_r

            # Select an action
            if policy_type == 'softmax':
                next_a = select_a_with_softmax(next_s, q_value, beta=beta)
            elif policy_type == 'epsilon_greedy':
                next_a = select_a_with_epsilon_greedy(next_s, q_value, epsilon=epsilon)
            else:
                raise ValueError("Invalid policy_type: {}".format(policy_type))

            # Calculation of TD error
            if algorithm_type == 'sarsa':
                delta = r + gamma * q_value[next_s, next_a] - q_value[curr_s, curr_a]
            elif algorithm_type == 'q_learning':
                delta = r + gamma * np.max(q_value[next_s, :]) - q_value[curr_s, curr_a]
            else:
                raise ValueError("Invalid algorithm_type: {}".format(algorithm_type))

            # Update a Q value table
            q_value[curr_s, curr_a] += alpha * delta

            curr_s = next_s
            curr_a = next_a

            if done:
                if policy_type == 'softmax':
                    print "Episode: {0}\t Steps: {1:>4}\tCumuR: {2:>5.2f}\tTermR: {3}\tBeta: {4:.3f}".format(i_episode, i_step, cumu_r, r, beta)
                    history.append([i_episode, i_step, cumu_r, r, beta])
                elif policy_type == 'epsilon_greedy':                
                    print "Episode: {0}\t Steps: {1:>4}\tCumuR: {2:>5.2f}\tTermR: {3}\tEpsilon: {4:.3f}".format(i_episode, i_step, cumu_r, r, epsilon)
                    history.append([i_episode, i_step, cumu_r, r, epsilon])
                else:
                    raise ValueError("Invalid policy_type: {}".format(policy_type))

                break

        if policy_type == 'epsilon_greedy':
            # epsilon is decayed expolentially
            epsilon = epsilon * epsilon_decay
        elif policy_type == 'softmax':
            # beta is increased linearly
            beta = init_beta + i_episode * beta_inc

    # Stop monitoring the simulation for OpenAI Gym
    # env.monitor.close()

    history = np.array(history)

    window_size = 100
    def running_average(x, window_size, mode='valid'):
        return np.convolve(x, np.ones(window_size)/window_size, mode=mode)

    # fig, ax = plt.subplots(2, 2, figsize=[12, 8])
    # # Number of steps
    # ax[0, 0].plot(history[:, 0], history[:, 1], '.') 
    # ax[0, 0].set_xlabel('Episode')
    # ax[0, 0].set_ylabel('Number of steps')
    # ax[0, 0].plot(history[window_size-1:, 0], running_average(history[:, 1], window_size))
    # # Cumulative reward
    # ax[0, 1].plot(history[:, 0], history[:, 2], '.') 
    # ax[0, 1].set_xlabel('Episode')
    # ax[0, 1].set_ylabel('Cumulative rewards')
    # ax[0, 1].plot(history[window_size-1:, 0], running_average(history[:, 2], window_size))
    # # Terminal reward
    # ax[1, 0].plot(history[:, 0], history[:, 3], '.') 
    # ax[1, 0].set_xlabel('Episode')
    # ax[1, 0].set_ylabel('Terminal rewards')
    # ax[1, 0].plot(history[window_size-1:, 0], running_average(history[:, 3], window_size))
    # # Epsilon/Beta
    # ax[1, 1].plot(history[:, 0], history[:, 4], '.') 
    # ax[1, 1].set_xlabel('Episode')
    # if policy_type == 'softmax':
    #     ax[1, 1].set_ylabel('Beta')
    # elif policy_type == 'epsilon_greedy':
    #     ax[1, 1].set_ylabel('Epsilon')
    # fig.savefig('./'+result_dir+'.png')

    print "Q value table:"
    print q_value

    if policy_type == 'softmax':
        print "Action selection probability:"
        print np.array([softmax(q, beta=beta) for q in q_value])
    elif policy_type == 'epsilon_greedy':
        print "Greedy action"
        greedy_action = np.zeros([n_s, n_a])
        greedy_action[np.arange(n_s), np.argmax(q_value, axis=1)] = 1
        #print np.array([zero_vec[np.argmax(q)] = 1 for q in q_value])
        print greedy_action

if __name__ == "__main__":
    main()

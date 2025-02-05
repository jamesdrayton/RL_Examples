import numpy as np
import time
import os
import matplotlib.pyplot as plt
import gymnasium
import gymnasium_env

# create CliffWalker environment using gymnasium
env = gymnasium.make("gymnasium_env/CliffWalker-v0", render_mode="human")

# QTable : contains the Q-Values for every (state,action) pair
numstates = (env.observation_space['agent'].high[0]+1) * (env.observation_space['agent'].high[1]+1)
numactions = env.action_space.n
qtable = np.random.rand(numstates, numactions).tolist()

# CHANGE: initialize a value for the total reward after all episodes
acc = 0

# hyperparameters
epochs = 50
gamma = 0.1
epsilon = 0.08
decay = 0.1
alpha = 1 # CHANGE: alpha to 1

# training loop
for i in range(epochs):
    state_dict, info = env.reset()
    state = (state_dict['agent'][0]+1)* (state_dict['agent'][1]+1) - 1
    # CHANGE: reset does not include a done value or cliff value
    cliff = False
    done = False
    steps = 0

    while not done:
        env.render()
        time.sleep(0.05)

        # count steps to finish game
        steps += 1

        # act randomly sometimes to allow exploration
        if np.random.uniform() < epsilon:
            action = env.randomAction()
        # if not select max action in Qtable (act greedy)
        else:
            action = qtable[state].index(max(qtable[state]))

        # CHANGE: uses cliff instead of truncated, but does the same thing
        # take action
        next_state_dict, reward, done, cliff, info = env.step(action)
        next_state = (next_state_dict['agent'][0]+1)* (next_state_dict['agent'][1]+1) - 1
        # CHANGE: update reward value two different ways
        # first  is applicable if cliff was considered in the reward in env
        # reward -= steps
        # second is applicable if cliff is not considered in the reward in env
        if cliff:
            reward -= 100
            # if episode does not end if cliff
            # env.reset()
        else:
            reward -= 1
        # CHANGE: add in alpha even though it would have no impact?
        # update qtable value with Bellman equation 
        qtable[state][action] = qtable[state][action] + (alpha * ((reward + (gamma * max(qtable[next_state]))) - qtable[state][action]))

        # update state
        state = next_state
    # The more we learn, the less we take random actions
    epsilon -= decay*epsilon

    print("\nDone in", steps, "steps".format(steps))
    print("Episode Reward: " + str(reward - steps)) # CHANGE: print the reward for the episode
    acc += (reward - steps)
    time.sleep(0.8)
#TODO: plot the cumulative reward over the episodes and steps per episode using matplotlib


env.close()
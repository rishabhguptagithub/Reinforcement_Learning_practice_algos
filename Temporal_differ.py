import sys
import gym
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt
%matplotlib inline

#import check_test
from plot_utils import plot_values

env = gym.make('CliffWalking-v0')

#-------------sarsa max--------------------#
def generate_episode_and_update_Q(env, Q, epsilon, nA,gamma,alpha):
    """ generates an episode from following the epsilon-greedy policy """
    episode = []
    state = env.reset()
    action = np.random.choice(np.arange(nA), p=get_probs(Q[state], epsilon, nA)) \
                                    if state in Q else env.action_space.sample()
    score=0
    while True:

#         print(state)
#         print("==========",Q[state])


        next_state, reward, done, info = env.step(action)
        next_action=np.random.choice(np.arange(nA), p=get_probs(Q[next_state], epsilon, nA)) \
                                    if state in Q else env.action_space.sample()
        score=score+reward

        old_Q=Q[state][action]
      #  Q[state][action] = old_Q + alpha*(reward+gamma*max(Q[next_state]) - old_Q)
        Q[state][action] = old_Q + alpha*(reward+gamma*(Q[next_state][next_action])-old_Q)
        state = next_state
        action=next_action
        if done:

            break
    return Q,score
def get_probs(Q_s, epsilon, nA):
    """ obtains the action probabilities corresponding to epsilon-greedy policy """
    policy_s = np.ones(nA) * epsilon / nA
    best_a = np.argmax(Q_s)
    policy_s[best_a] = 1 - epsilon + (epsilon / nA)
    return policy_s


def sarsa(env, num_episodes, alpha, gamma=1.0,eps_start=1,eps_min=0.8):
    # initialize action-value function (empty dictionary of arrays)
    Q = defaultdict(lambda: np.zeros(env.nA))
    # initialize performance monitor
    # loop over episodes

    tmp_scores = deque(maxlen=200)     # deque for keeping track of scores
    avg_scores = deque(maxlen=num_episodes)   # average scores over every plot_every episodes

    epsilon=eps_start
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 20 == 0:
            print("------------------")
            print("\rEpisode {}/{}".format(i_episode, num_episodes))
            sys.stdout.flush()
        epsilon=max(epsilon*0.9999,eps_min)

        Q,score=generate_episode_and_update_Q(env,Q,epsilon,4,gamma,alpha)
        tmp_scores.append(score)
        if(i_episode%200==0):
            avg_scores.append(np.mean(tmp_scores))


    plt.plot(np.linspace(0,num_episodes,len(avg_scores),endpoint=False), np.asarray(avg_scores))
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
    plt.show()
    # print best 100-episode performance
    print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(avg_scores))

        ## TODO: complete the function

    return Q

Q_sarsa = sarsa(env, 2000, .01)

# print the estimated optimal policy
policy_sarsa = np.array([np.argmax(Q_sarsa[key]) if key in Q_sarsa else -1 for key in np.arange(48)]).reshape(4,12)
#check_test.run_check('td_control_check', policy_sarsa)
print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
print(policy_sarsa)

# plot the estimated optimal state-value function
V_sarsa = ([np.max(Q_sarsa[key]) if key in Q_sarsa else 0 for key in np.arange(48)])
plot_values(V_sarsa)




#----------------Q Learning -------------------#


def generate_episode_and_update_Q(env, Q, epsilon, nA,gamma,alpha):
    """ generates an episode from following the epsilon-greedy policy """
    episode = []
    state = env.reset()
    action = np.random.choice(np.arange(nA), p=get_probs(Q[state], epsilon, nA)) \
                                    if state in Q else env.action_space.sample()
    score=0
    while True:

#         print(state)
#         print("==========",Q[state])


        next_state, reward, done, info = env.step(action)
        next_action=np.random.choice(np.arange(nA), p=get_probs(Q[next_state], epsilon, nA)) \
                                    if state in Q else env.action_space.sample()
        score=score+reward

        old_Q=Q[state][action]
        Q[state][action] = old_Q + alpha*(reward+gamma*max(Q[next_state]) - old_Q)
        #Q[state][action] = old_Q + alpha*(reward+gamma*(Q[next_state][next_action])-old_Q)
        state = next_state
        action=next_action
        if done:

            break
    return Q,score
def get_probs(Q_s, epsilon, nA):
    """ obtains the action probabilities corresponding to epsilon-greedy policy """
    policy_s = np.ones(nA) * epsilon / nA
    best_a = np.argmax(Q_s)
    policy_s[best_a] = 1 - epsilon + (epsilon / nA)
    return policy_s


def Q_learning(env, num_episodes, alpha, gamma=1.0,eps_start=1,eps_min=0.8):
    # initialize action-value function (empty dictionary of arrays)
    Q = defaultdict(lambda: np.zeros(env.nA))
    # initialize performance monitor
    # loop over episodes

    tmp_scores = deque(maxlen=200)     # deque for keeping track of scores
    avg_scores = deque(maxlen=num_episodes)   # average scores over every plot_every episodes

    epsilon=eps_start
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 20 == 0:
            print("------------------")
            print("\rEpisode {}/{}".format(i_episode, num_episodes))
            sys.stdout.flush()
        epsilon=max(epsilon*0.9999,eps_min)

        Q,score=generate_episode_and_update_Q(env,Q,epsilon,4,gamma,alpha)
        tmp_scores.append(score)
        if(i_episode%200==0):
            avg_scores.append(np.mean(tmp_scores))


    plt.plot(np.linspace(0,num_episodes,len(avg_scores),endpoint=False), np.asarray(avg_scores))
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
    plt.show()
    # print best 100-episode performance
    print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(avg_scores))

        ## TODO: complete the function

    return Q


# obtain the estimated optimal policy and corresponding action-value function
Q_sarsamax = Q_learning(env, 5000, .01)

# print the estimated optimal policy
policy_sarsamax = np.array([np.argmax(Q_sarsamax[key]) if key in Q_sarsamax else -1 for key in np.arange(48)]).reshape((4,12))
#check_test.run_check('td_control_check', policy_sarsamax)
print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
print(policy_sarsamax)

# plot the estimated optimal state-value function
plot_values([np.max(Q_sarsamax[key]) if key in Q_sarsamax else 0 for key in np.arange(48)])



#------------------------expected sarsa--------------------------#

def generate_episode_and_update_Q(env, Q, epsilon, nA,gamma,alpha):
    """ generates an episode from following the epsilon-greedy policy """
    episode = []
    state = env.reset()
    action = np.random.choice(np.arange(nA), p=get_probs(Q[state], epsilon, nA)) \
                                    if state in Q else env.action_space.sample()
    score=0
    while True:

#         print(state)
#         print("==========",Q[state])


        next_state, reward, done, info = env.step(action)
        next_action=np.random.choice(np.arange(nA), p=get_probs(Q[next_state], epsilon, nA)) \
                                    if state in Q else env.action_space.sample()
        score=score+reward

        old_Q=Q[state][action]
        t=0
        p=get_probs(Q[next_state],epsilon,nA)
        for a in range(nA):
            t=t+(p[a]*Q[next_state][a])
#             c=Q[next_state][next_action]
#             if c== np.max(Q[next_state]):
#                 t=t+np.max(p)*c
#             else:
#                 t=t+p

        Q[state][action] = old_Q + alpha*(reward+gamma*t - old_Q)
        #Q[state][action] = old_Q + alpha*(reward+gamma*(Q[next_state][next_action])-old_Q)
        state = next_state
        action=next_action
        if done:

            break
    return Q,score
def get_probs(Q_s, epsilon, nA):
    """ obtains the action probabilities corresponding to epsilon-greedy policy """
    policy_s = np.ones(nA) * epsilon / nA
    best_a = np.argmax(Q_s)
    policy_s[best_a] = 1 - epsilon + (epsilon / nA)
    return policy_s


def expected_sarsa(env, num_episodes, alpha, gamma=1.0,eps_start=1,eps_min=0.8):
    # initialize action-value function (empty dictionary of arrays)
    Q = defaultdict(lambda: np.zeros(env.nA))
    # initialize performance monitor
    # loop over episodes

    tmp_scores = deque(maxlen=200)     # deque for keeping track of scores
    avg_scores = deque(maxlen=num_episodes)   # average scores over every plot_every episodes

    epsilon=eps_start
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 20 == 0:
            print("------------------")
            print("\rEpisode {}/{}".format(i_episode, num_episodes))
            sys.stdout.flush()
        epsilon=max(epsilon*0.9999,eps_min)

        Q,score=generate_episode_and_update_Q(env,Q,epsilon,4,gamma,alpha)
        tmp_scores.append(score)
        if(i_episode%200==0):
            avg_scores.append(np.mean(tmp_scores))


    plt.plot(np.linspace(0,num_episodes,len(avg_scores),endpoint=False), np.asarray(avg_scores))
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
    plt.show()
    # print best 100-episode performance
    print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(avg_scores))

        ## TODO: complete the function

    return Q


Q_expsarsa = expected_sarsa(env, 10000, 1)

# print the estimated optimal policy
policy_expsarsa = np.array([np.argmax(Q_expsarsa[key]) if key in Q_expsarsa else -1 for key in np.arange(48)]).reshape(4,12)
check_test.run_check('td_control_check', policy_expsarsa)
print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
print(policy_expsarsa)

# plot the estimated optimal state-value function
plot_values([np.max(Q_expsarsa[key]) if key in Q_expsarsa else 0 for key in np.arange(48)])

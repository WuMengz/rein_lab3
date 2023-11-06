# -*- coding: utf-8 -*-

import numpy as np
import gym
import matplotlib.pyplot as plt
from math import pow

env = gym.make("FrozenLake-v1")  # 创建环境
env.reset()

def plot_moving_average(data, type = "", title = "", k = 50):
    # print(data)
    presum = 0
    datalen = len(data)
    for i in range(datalen):
        presum += data[i]
        data[i] = presum
    for i in range(datalen - 1, k - 1, -1):
        data[i] = data[i] - data[i - k]
    for i in range(datalen):
        data[i] /= k if i >= k else i + 1
    plt.plot(data)
    plt.xlabel("episodes(*100)")
    plt.ylabel("moving average of " + type)
    plt.title("SARSA(lambda=1.0)")
    plt.show()

def Getloss(Q, env, gamma):
    loss = 0
    for s in range(env.observation_space.n):
        for a in range(env.action_space.n):
            target = 0
            for prob, next_state, reward, done in env.P[s][a]:
                next_action = np.argmax(Q[next_state, :])
                target += prob * (reward + gamma * Q[next_state, next_action])
            loss += pow(Q[s, a] - target, 2)
    return loss

def sarsa(env, num_episodes=1000, alpha=0.1, gamma=0.99, eps0=1, decay=0.001):
    # 初始化Q表为0
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    td_loss = []
    test_reward = []
    # 针对每个回合进行更新
    for i_episode in range(num_episodes):
        # 初始化状态
        state = env.reset()
        # 使用epsilon-greedy策略选择动作
        epsilon = eps0 / (1 + decay * i_episode)
        if i_episode % 100 == 0:
            print(f"iteration: {i_episode}, epsilon: {epsilon}")
            td_loss.append(Getloss(Q, env, gamma))
            test_reward.append(test_pi(env, np.argmax(Q, axis=1), num_episodes=100))
        action = epsilon_greedy(Q, state, env.action_space.n, epsilon)
        # 针对每个时间步进行更新
        while True:
            # 执行选定的动作
            next_state,reward,done,_= env.step(action)
            # 使用epsilon-greedy策略选择下一个动作
            next_action = epsilon_greedy(Q, next_state, env.action_space.n, epsilon)
            # 计算TD误差
            td_target = reward + gamma * Q[next_state, next_action] * (not done)
            td_error = td_target-Q[state, action]
            # 更新Q表
            Q[state, action] += alpha * td_error
            # 更新状态和动作
            state = next_state
            action = next_action
            if done:
                break

    policy = np.argmax(Q, axis=1)
    plot_moving_average(td_loss, type = "td_loss")
    plot_moving_average(test_reward, type = "reward")

    # 返回最终的Q表和策略
    return Q, policy

def sarsa_nstep(env, nstep, num_episodes=1000, alpha=0.1, gamma=0.99, eps0=1, decay=0.001):
    # 初始化Q表为0
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    td_loss = []
    test_reward = []
    # 针对每个回合进行更新
    for i_episode in range(num_episodes):
        # 初始化状态
        state = env.reset()
        # 使用epsilon-greedy策略选择动作
        epsilon = eps0 / (1 + decay * i_episode)
        if i_episode % 100 == 0:
            print(f"iteration: {i_episode}, epsilon: {epsilon}")
            td_loss.append(Getloss(Q, env, gamma))
            test_reward.append(test_pi(env, np.argmax(Q, axis=1), num_episodes=100))

        action = epsilon_greedy(Q, state, env.action_space.n, epsilon)
        states = [state]
        actions = [action]
        rewards = []
        # 针对每个时间步进行更新
        while True:
            # 执行选定的动作
            next_state,reward,done,_= env.step(action)
            # 使用epsilon-greedy策略选择下一个动作
            next_action = epsilon_greedy(Q, next_state, env.action_space.n, epsilon)
            # 计算TD误差
            # td_target = reward + gamma * Q[next_state, next_action] * (not done)
            # td_error = td_target-Q[state, action]n
            # 更新Q表
            # Q[state, action] += alpha * td_error
            # 更新状态和动作
            state = next_state
            action = next_action
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            if done:
                break
        
        for i in range(len(states) - 1):
            state = states[i]
            action = actions[i]

            last_idx = min(len(states) - 1, i + nstep)
            td_target = Q[states[last_idx], actions[last_idx]]
            for j in reversed(range(i, last_idx)):
                td_target = td_target * gamma + rewards[j]
            td_error = td_target - Q[state, action]
            
            Q[state, action] += alpha * td_error
    
    policy = np.argmax(Q, axis=1)
    plot_moving_average(td_loss, type = "td_loss")
    plot_moving_average(test_reward, type = "reward")

    # 返回最终的Q表和策略
    return Q, policy

def sarsa_lambda(env, lambd, num_episodes=1000, alpha=0.1, gamma=0.99, eps0=1, decay=0.001):
    # 初始化Q表为0
    state_n = env.observation_space.n
    action_n = env.action_space.n
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    td_loss = []
    test_reward = []
    # 针对每个回合进行更新
    for i_episode in range(num_episodes):
        # 初始化状态
        state = env.reset()
        # 使用epsilon-greedy策略选择动作
        epsilon = eps0 / (1 + decay * i_episode)
        if i_episode % 100 == 0:
            print(f"iteration: {i_episode}, epsilon: {epsilon}")
            td_loss.append(Getloss(Q, env, gamma))
            test_reward.append(test_pi(env, np.argmax(Q, axis=1), num_episodes=100))

        action = epsilon_greedy(Q, state, env.action_space.n, epsilon)
        E = np.zeros((env.observation_space.n, env.action_space.n))
        # 针对每个时间步进行更新
        while True:
            # 执行选定的动作
            next_state,reward,done,_= env.step(action)
            # 使用epsilon-greedy策略选择下一个动作
            next_action = epsilon_greedy(Q, next_state, env.action_space.n, epsilon)
            for i in range(state_n):
                for j in range(action_n):
                    E[i, j] = E[i, j] * lambd * gamma
            E[state, action] += 1
            # 计算TD误差
            td_target = reward + gamma * Q[next_state, next_action] * (not done)
            td_error = td_target - Q[state, action]
            # 更新Q表
            for i in range(state_n):
                for j in range(action_n):
                    Q[i, j] += alpha * td_error * E[i, j]
            # 更新状态和动作
            state = next_state
            action = next_action
            if done:
                break
    
    policy = np.argmax(Q, axis=1)
    plot_moving_average(td_loss, type = "td_loss")
    plot_moving_average(test_reward, type = "reward")

    # 返回最终的Q表和策略
    return Q, policy

def epsilon_greedy(Q, state, num_actions, epsilon):
    if np.random.random() < epsilon:
        # 随机选择动作
        return np.random.choice(num_actions)
    else:
        # 选择最优动作
        return np.argmax(Q[state, :])

def test_pi(env, pi, num_episodes=1000):
   """
    测试策略。
    参数：
    env -- OpenAI Gym环境对象。
    pi -- 需要测试的策略。
    num_episodes -- 进行测试的回合数。

    返回值：
    成功到达终点的频率。
    """

   count = 0
   for e in range(num_episodes):
        ob = env.reset()
        for t in range(1000):
            a = pi[ob]
            ob, rew, done, _ = env.step(a)
            if done:
                count += 1 if rew == 1 else 0
                break
   return count / num_episodes

# Q, pi = sarsa(env, num_episodes=50000)
# Q, pi = sarsa_nstep(env, 1, num_episodes=500000, decay=0.0001)
Q, pi = sarsa_lambda(env, 1, num_episodes=50000)
result = test_pi(env, pi)
print(result)

# -*- coding: utf-8 -*-

import numpy as np
import gym

env = gym.make("FrozenLake-v0")  # 创建环境
env.reset()

def sarsa(env, num_episodes=1000, alpha=0.1, gamma=0.99, eps0=1, decay=0.001):
    # 初始化Q表为0
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    # 针对每个回合进行更新
    for i_episode in range(num_episodes):
        # 初始化状态
        state = env.reset()
        # 使用epsilon-greedy策略选择动作
        epsilon = eps0 / (1 + decay * i_episode)
        print(f"iteration: {i_episode}, epsilon: {epsilon}")
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
    # 返回最终的Q表和策略
    policy = np.argmax(Q, axis=1)
    return Q, policy

def epsilon_greedy(Q, state, num_actions, epsilon):
    if np.random.random() < epsilon:
        # 随机选择动作
        return np.random.choice(num_actions)
    else:
        # 选择最优动作
        return np.argmax(Q[state, :])

def test_pi(env, pi, num_episodes=100):
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
        for t in range(100):
            a = pi[ob]
            ob, rew, done, _ = env.step(a)
            if done:
                count += 1 if rew == 1 else 0
                break
   return count / num_episodes

Q, pi = sarsa(env, num_episodes=10000)
result = test_pi(env, pi)
print(result)

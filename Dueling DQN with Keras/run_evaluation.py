"""
Dueling DQN & Natural DQN comparison

Lin Cheng 2018.01.15

"""

## package input
import gym
import numpy as np
from DQN_method_keras import Dueling_DQN_method
import matplotlib.pyplot as plt

# 导入environment
env = gym.make('Pendulum-v0')
env = env.unwrapped
env.seed(1)

# 确定动作和状态维度
action_dim = 25
action_high=env.action_space.high[0]
action_low=env.action_space.low[0]
action_sequence = np.linspace(action_low, action_high, action_dim)
state_dim = env.observation_space.shape[0]
print(action_sequence)
print(state_dim)

#

Dueling_DQN = Dueling_DQN_method(action_dim, state_dim, reload_flag=True)


def train(RL):
    acc_r = [0]
    total_step = 0
    for ep in range(10):
        state_now = env.reset()
        state_now = np.reshape(state_now, [1, 3])
        for step in range(2000):
            env.render()

            # 动作选取
            action_index = RL.chose_action(state_now, train=False)
            # action_index = np.random.randint(0, action_dim)
            action = action_sequence[action_index]
            # print(action_index)

            # 状态返回
            state_next, reward, done, info = env.step(np.array([action]))
            state_next = np.reshape(state_next, [1, 3])
            reward /= 10  #收益函数
            # acc_r.append(reward + RL.gamma*acc_r[-1])
            acc_r.append(reward + acc_r[-1])

            state_now = state_next
            total_step += 1
        print(ep)
    return 1

Q_dueling = train(Dueling_DQN)















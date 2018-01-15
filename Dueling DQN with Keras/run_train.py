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

Dueling_DQN = Dueling_DQN_method(action_dim, state_dim, reload_flag=False)


def train(RL):
    acc_r = [0]
    total_step = 0
    state_now = env.reset()
    state_now = np.reshape(state_now, [1, 3])
    while True:
        if total_step > RL.memory_size + 9000: env.render()

        # 动作选取
        action_index = RL.chose_action(state_now, train=True)
        # action_index = np.random.randint(0, action_dim)
        action = action_sequence[action_index]

        # 状态返回
        state_next, reward, done, info = env.step(np.array([action]))
        state_next = np.reshape(state_next, [1, 3])
        reward /= 10  #收益函数
        # acc_r.append(reward + RL.gamma*acc_r[-1])
        acc_r.append(reward + acc_r[-1])

        # store memory
        RL.memory_store(state_now, action_index, reward, state_next, done)

        if total_step > RL.memory_size:
            RL.Learn()

        if total_step - RL.memory_size > 35000:
            break

        state_now = state_next
        total_step += 1

    RL.model_save()
    return acc_r

Q_dueling = train(Dueling_DQN)


plt.figure(1)
plt.plot(np.array(Q_dueling))
plt.grid()
plt.show()














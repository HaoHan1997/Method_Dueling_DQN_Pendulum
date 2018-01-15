
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
from keras.models import load_model

class DDQN_method:

    def __init__(self, n_actions, n_feature):
        self.action_dim = n_actions
        self.state_dim = n_feature
        self.memory_counter = 0
        self.memory_size = 1000
        self.memory = np.empty([self.memory_size, 11])
        self.batch_size = 100
        self.gamma = 0.9
        self.epsilon = 1
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.1
        self.learning_rate = 0.01
        self.learn_step_counter = 0
        self.replace_target_limit = 100
        self.build_model()

    def build_model(self):

        activation_curve = 'relu'
        unit_num = 50
        # eval network
        model = Sequential()
        model.add(Dense(batch_input_shape=[None, self.state_dim], units=unit_num, activation=activation_curve))
        model.add(Dense(units=unit_num, activation=activation_curve))
        # model.add(Dense(units=128, activation = activation_curve))
        model.add(Dense(units=3, activation='linear'))
        model.compile(loss='myloss_dueling', optimizer=RMSprop(lr=self.learning_rate), )
        # model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate),)
        self.model_eval = model

        # target network
        model2 = Sequential()
        model2.add(Dense(batch_input_shape=[None, self.state_dim], units=unit_num, activation=activation_curve))
        model2.add(Dense(units=unit_num, activation=activation_curve))
        # model2.add(Dense(units=128, activation=activation_curve))
        model2.add(Dense(units=3, activation='linear'))
        model2.compile(loss='myloss_dueling', optimizer=RMSprop(lr=self.learning_rate), )
        self.model_target = model2


    def memory_store(self, state_now, action, reward, state_next, done):

        action = np.reshape(action, [1, 1])
        reward = np.reshape(reward, [1, 1])
        done = np.reshape(done, [1, 1])
        transition = np.hstack((state_now, action, reward, state_next, done))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def chose_action(self, state, train=True):

        if train and (np.random.uniform() < self.epsilon):
            # action = self.action_space.sample()
            action = np.random.randint(0, self.action_dim)
        else:
            q_eval = self.model_eval.predict(state)
            action = np.argmax(q_eval)

        return action



    def Learn(self):

        if self.learn_step_counter % self.replace_target_limit == 0:
            eval_weights = self.model_eval.get_weights()
            self.model_target.set_weights(eval_weights)
            print('\ntarget_params_replaced\n')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            print('memory is not full')

        batch_memory = self.memory[sample_index, :]

        batch_state = batch_memory[:, :self.state_dim]
        batch_action = batch_memory[:, self.state_dim].astype(int)
        batch_reward = batch_memory[:, self.state_dim+1]
        batch_state_next = batch_memory[:, -self.state_dim-1:-1]
        batch_done = batch_memory[:, -1]

        q_target = self.model_output_trans(self.model_eval.predict(batch_state))
        q_next1 = self.model_output_trans(self.model_eval.predict(batch_state_next))
        q_next2 = self.model_output_trans(self.model_target.predict(batch_state_next))
        batch_action_withMaxQ =  np.argmax(q_next1, axis=1)
        batch_index11 = np.arange(self.batch_size, dtype=np.int32)
        q_next_Max = q_next2[batch_index11, batch_action_withMaxQ]
        q_target[batch_index11, batch_action] = batch_reward + (1-batch_done)*self.gamma * q_next_Max

        self.model_eval.fit(batch_state, q_target, verbose=0)
        self.learn_step_counter += 1

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def data_save(self):
        # model save
        self.model_eval.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'

    def data_load(self):

        # model load
        self.model_eval = load_model('my_model.h5')




































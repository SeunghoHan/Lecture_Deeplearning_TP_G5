from __future__ import division

import numpy as np
import random
import tensorflow as tf
import os

from gridworld.py import gameEnv

class Agent():
    def __init__(self,hiddens = [256, 256, 256],num_frames = 4, state_size = 10,
                 action_size = 6, env, lr = 1e-3, batch_size = 32, num_episodes = 10000,
                 buffer_size = 50000, start_e = 1, final_e = 0.01, gamma = 0.99,
                 update_freq = 5, load_model = False , logs_dir = "./logs_dir"):

        self.hiddens = hiddens # Size of the hidden layers
        self.num_frames = frames # Number of consecutive state frames
        self.state_size = state_size # Size of the state vector
        self.action_size = action_size  # Number of actions
        self.env = env # Save the environmet!
        self.lr=lr # learning rate of the optimizer
        self.batch_size=batch_size # Size of the experience sample batch
        self.num_episodes = num_episodes # Number of game environmet episodes in which we train
        self.max_ep_length = max_ep_length # Maximun length (number of actions) of a single train episode
        self.buffer_size = buffer_size # Size of the experience replay buffer
        self.start_e = start_e # Inital value of the exploration coefficient
        self.final_e = final_e # Final value of the exploration coefficient
        self.gamma = gamma # Discount factor on the target Q-value
        self.tau = tau # Porcentage that determines how much are parameters of mainQN net modified by targetQN
        self.update_freq = update_freq # Frecuency of updates of the double DQN
        self.logs_dir = logs_dir # Path to store logs and checkpoints

        tf.reset_default_graph()

        # Instantiate the networks
        self.mainQN = DQN("mainQN", state_size, action_size, hiddens, num_frames)
        self.targetQN = DQN("targetQN", state_size, action_size, hiddens, num_frames)

        init = tf.global_variables_initializer()
        trainables = tf.trainable_variables()

        self.target_ops = set_target_graph_vars(trainables, tau)

        # Create a experience replay buffer & score records
        self.exp_buffer = ExperienceBuffer()
        self.step_record = []
        self.reward_record = []

    def learn(self):

        init = tf.global_variables_initializer()
        sess = tf.Session()
        e = self.start_e
        current_step = 0 # previously total_steps

        self.model_saver = ModelSaver(self.logs_dir)

        with sess:

            sess.run(init)
            if self.load_model == True:
                print('Loading Model...')
                ckpt = model_saver_restore_model(sess)

            # Set the target network to be equal to the primary network
            update_target_graph(self.target_ops, sess)

            # Start the pre train proces
            for episode in range(self.num_episodes):

                if episode % 100 == 0:
                    print("\n=====" + "Episode " + str(episode) + "starts =====" )

                episode_exp = ExperienceBuffer()

                #Reset environment and get first new observation
                s = env.reset()

                d = False # episode's "done" signal
                episode_reward_sum = 0
                episode_steps = 0

                #The Q-Network
                while episode_steps < max_ep_length: #If the agent take too long to win, end the trial.
                    episode_steps += 1

                    #Choose an action by greedily (with e chance of random action) from the Q-network
                    if np.random.rand(1) < e or current_step < self.pre_train_steps:
                        a = np.random.randint(0,4)
                    else:
                        a = sess.run(self.mainQN.predict,feed_dict={mainQN.state:[s]})[0]

                    s1,r,d = env.step(a)
                    current_step += 1
                    episode_exp.add(np.reshape(np.array([s,a,r,s1,d]),[1,5])) # Save the experience to our episode buffer.

                    # Start train process
                    if current_step > self.pre_train_steps:

                        if e > self.final_e:
                            stepDrop = 1/10000
                            e -= stepDrop

                        if current_step % self.update_freq == 0:

                            train_batch = self.exp_buffer.sample(self.bach_size) #Get a random batch of experiences.

                            #Perform the Double-DQN update to the target Q-values
                            Q1 = sess.run(self.mainQN.predict,
                                          feed_dict={self.mainQN.state:np.vstack(train_batch[:,3])})

                            Q2 = sess.run(self.targetQN.Qout,
                                          feed_dict={self.targetQN.state:np.vstack(train_batch[:,3])})

                            end_multiplier = -(train_batch[:,4] - 1)
                            doubleQ = Q2[range(self.batch_size),Q1]
                            targetQ = train_batch[:,2] + (self.gamma*doubleQ*end_multiplier)

                            # Update the network with our target values.
                            _ = sess.run(self.mainQN.updateModel,
                                         feed_dict={self.mainQN.state:np.vstack(train_batch[:,0]),
                                         self.mainQN.targetQ:targetQ,
                                         self.mainQN.actions:train_batch[:,1]})

                            # Set the target network to be equal to the primary
                            self.update_target_graph(self.target_ops, sess)

                    episode_reward_sum += r
                    s = s1

                    if d == True:
                        break

                self.exp_buffer.add(episode_exp.buffer)
                self.step_record.append(episode_steps)
                self.reward_record.append(episode_reward_sum)

                #Periodically save the model.
                if episode % 1000 == 0:
                    self.model_saver.save_model(sess)
                    print("Model save_model")

                if len(self.reward_record) % 10 == 0:
                    print(current_step, np.mean(self.reward_record[-10:]), e)

            self.model_saver.save_model(sess)

        print("Percent of succesful episodes: " + str(100*sum(self.reward_record)/self.num_episodes) + "%")

    def test(self):


    ''' Auxiliary Methods '''
    # Originally called updateTargetGraph
    def set_target_graph_vars(self, tfVars, tau):
        total_vars = len(tfVars)
        op_holder = []

        for idx,var in enumerate(tfVars[0:total_vars//2]): # Select the first half of the variables (mainQ net)
            op_holder.append( tfVars[idx+total_vars//2].assign((var.value()*tau)+((1-tau)*tfVars[idx+total_vars//2].value())))

        return op_holder

    # Originally called updateTarget
    def update_target_graph(op_holder, sess):
        for op in op_holder:
            sess.run(op)

class DQN():
    def __init__(self, net_name, state_size, action_size, hiddens, num_Frames):
        self.net_name = net_name

        with tf.variable_scope(self.net_name):

            self.state = tf.placeholder(shape=[None, num_frames, state_size], dtype=tf.float32)
            self.inputState = tf.reshape(state, [-1, num_frames * action_size])

            # Weights of each layer
            self.W = {
                'W1': self.init_weight("W1", [num_frames * action_size, hiddens[0]]),
                'W2': self.init_weight("W2", [hiddens[0], hiddens[1]]),
                'W3': self.init_weight("W3", [hiddens[1], hiddens[2]]),
                'AW': self.init_weight("AW", [hiddens[2]//2, hiddens[2]]),
                'VM': self.init_weight("AW", [hiddens[2]//2, 1])
            }

            self.b = {
                'b2': self.init_bias([hiddens[0]]),
                'b1': self.init_bias([hiddens[1]]),
                'b3': self.init_bias([hiddens[2]]),
            }

            # Layers
            self.hidden1 = tf.nn.relu(tf.add(tf.matmul(action_size, self.W['W1']), self.b['b1']))
            self.hidden2 = tf.nn.relu(tf.add(tf.matmul(self.hidden1, self.W['W2']), self.b['b2']))
            self.hidden3 = tf.nn.relu(tf.add(tf.matmul(self.hidden2, self.W['W3']), self.b['b3']))

            # Compute the Advantage, Value, and total Q value
            self.A, self.V = tf.split(self.hidden3, 2, 1)
            self.Advantage = tf.matmul(self.A, self.AW)
            self.Value = tf.matmul(self.V, self.VM)
            self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))

            # Calcultate the action with highest Q value
            self.predict = tf.argmax(self.Qout, 1)

            # Compute the loss (sum of squared differences)
            self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_one_hot = tf.one_hot(self.actions, action_size, dtype=tf.float32)

            self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_one_hot), axis=1)
            self.td_error = tf.square(self.targetQ - self.Q)
            self.loss = tf.reduce_mean(self.td_error)

            self.trainer = tf.train.AdamOptimizer(learningRate)
            self.updateModel = self.trainer.minimize(self.loss)

    def init_weight(self, name, shape):
        return tf.get_variable(name=name, shape=shape,
                               initializer=tf.contrib.layers.xavier_initializer())

    def init_bias(self, name ,shape):
        return tf.get_variable(name=name, shape=shape,
                               initializer= tf.random_initializer())

class ExperienceBuffer():
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):

        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []

        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)),[size,5])

class ModelSaver():
    def __init__(self, path):
        self.saver = tf.train.Saver()
        self.ckptPath = path
        if not os.path.exists(path):
            os.makedirs(path)

    def restore_model(self, sess):
        ckpt = tf.train.get_checkpoint_state(self.ckptPath)
        self.saver.restore(sess, ckpt.model_checkpoint_path)

    def save_model(self, sess):
        self.saver.save(sess, self.ckptPath+'/model-'+str(i)+'.cptk')

def main():
    env = gameEnv(partial = False, gride_size=15)

    agent = Agent([256, 256, 256], 4, 10, 6, env, 1e-3, 32, 10000, 100, 50000,
                  1, 0.01, 10000, 0.99, 0.001, 5, False, "./log_dir"
                  )

    agent.learn()



if __name__ == '__main__':
    main()

from __future__ import division 

import gym
import numpy as np
import random
import tensorflow as tf 
import tensorflow.contrib.slim as slim
import scipy.misc
import os


class Agent():

    def __init__(self, stateSize = 10, actionSize = 6, numFrames = 4, 
                 batchSize = 32, hSize = 256, learningRate = 0.001, 
                 batchAccumulator, updateFreq = 5, y = 9,
                 numEpisodes = 10000, preTrainSteps = 10000, 
                 max_epLength = 500, tau = 0.001, lode_model = False, ckptPath = "./checkpoints"):
        
        self.stateSize=stateSize # Size of the state vector
        self.actionSize=actionSize # Number of actions
        self.numFrames=numFrames # Number of consecutive state frames
        self.batchSize=batchSize # Size of the experience sample 
        self.batchAccumulator = batchAccumulator # Operation for reward-over-time calculation
        self.hSize = hSize # Size of the hidden layers
        self.update_freq = update_freq # Frecuency of weight updates
        self.y = .99 # Discount factor on the target Q-values
        self.numEpisodes = numEpisodes # Number of game environmet episodes in which we train
        self.preTrainSteps = preTrainSteps #

        self.ckptPath = ckptPath
        if not os.path.exists(ckptPath):
            os.makedirs(path)

        tf.reset_default_graph()

        self.mainQN = self.build_network("mainQN", self.stateSize, self.actionSize, 
                                         self.hSize, self.numFrames, self.learningRate)
        
        self.targetQN = self.build_network("targetQN", self.stateSize, self.actionSize, 
                                           self.hSize, self.numFrames, self.learningRate)
        
        self.exp = Experience()
        self.stepRecord = []
        self.rewardRecord = []
        self.totalSteps = 0
       
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        self.trainables = tf.



    def build_network(self, netName = "mainQN", inputDim, outputDim, hiddenDim, 
                      numFrames, learningRate):

        with tf.variable_scope(self.netName):

            self.state = tf.placeholder(shape=[None, numFrames, inputDim], dtype=tf.float32)
            self.inputState = tf.reshape(state, [-1, numFrames * inputDim])

            # Weights of each layer
            self.W = {
                'W1': self.init_weight("W1", [numFrames * inputDim, hiddenDim]),
                'W2': self.init_weight("W2", [hiddenDim, hiddenDim]),
                'W3': self.init_weight("W3", [hiddenDim, hiddenDim]),
                'AW': self.init_weight("AW", [hiddenDim//2, hiddenDim]),
                'VM': self.init_weight("AW", [hiddenDim//2, 1])
            }

            self.hidden1 = tf.nn.relu(tf.matmul(self.inputState, self.W['W1']))
            self.hidden2 = tf.nn.relu(tf.matmul(self.hidden1, self.W['W2']))
            self.hidden3 = tf.nn.relu(tf.matmul(self.hidden2, self.W['W3']))
            
            '''
            # Uncomment this block to implement dropout 
            self.dropProb = 0.0
            self.hidden1 = tf.nn.dropout(self.hidden1, self.dropProb)
            self.hidden2 = tf.nn.dropout(self.hidden2, self.dropProb)
            self.hidden3 = tf.nn.dropout(self.hidden3, self.dropProb)
            '''

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
            self.actionsOneHot = tf.one_hot(self.actions, actionSize, dtype=tf.float32)

            self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actionsOneHot), axis=1)
            self.tdError = tf.square(self.targetQ - self.Q)
            self.loss = tf.reduce_mean(self.td_error)

            self.trainer = tf.train.AdamOptimizer(learningRate)
            self.updateModel = self.trainer.minimize(self.loss)  


    def init_weight(self, name, shape):
        return tf.get_variable(name=name, shape=shape, 
                               initializer=tf.contrib.layers.xavier_initializer())

    def update_target_graph(self, tfVars, tau):
        total_vars = len(tfVars)
        op_holder = []
    
        for idx,var in enumerate(tfVars[0:total_vars//2]): # Select the first half of the variables (mainQ net) 
            op_holder.append( tfVars[idx+total_vars//2].assign((var.value()*tau)+((1-tau)*tfVars[idx+total_vars//2].value()))) 
    
        return op_holder

    def get_q(self, state):

    def get_action(self, state, epsilon):

    def train(self):

    def test(self):    
 

class Experience():
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size
        
    def add(self, experience):
        
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        
        self.buffer.extend(experience)
        
    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)),[size,5])




        

     

def main():
    agent = Agent('''insert arguments here''')
    agent.train()
    agent.test()

if __name__ == '__main__':
    main()
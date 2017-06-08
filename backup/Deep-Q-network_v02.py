from __future__ import division 

import gym
import numpy as np
import random
import tensorflow as tf 
import tensorflow.contrib.slim as slim
import scipy.misc
import os


# Small grid for the nivation task (bird-eye sight)
#The game environment outputs 84x84x3 color images,
# and uses function calls as similar to the OpenAI gym
from gridworld_v02 import gameEnv
env = gameEnv(partial=False, size=10)

# --- Network implementation (conv) ---
class Qnetwork():
    def __init__(self, h_size): #// h_size is the number of activation of conv4 
        #The network recieves a frame from the game, flattened into an array.
        #It then resizes it and processes it through four convolutinoal layers.
        self.scalarInput = tf.placeholder(shape=[None,84*84*3], dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1,84,84,3])
        
        self.conv1 = slim.conv2d(inputs=self.imageIn,
                                 num_outputs=32,
                                 kernel_size=[8,8],
                                 stride=[4,4],
                                 padding='VALID', 
                                 biases_initializer = None)
        
        self.conv2 = slim.conv2d(inputs=self.conv1,
                                 num_outputs=64,
                                 kernel_size=[4,4],
                                 stride=[2,2],
                                 padding='VALID',
                                 biases_initializer = None)
        
        self.conv3 = slim.conv2d(inputs=self.conv2,
                                 num_outputs=64,
                                 kernel_size=[3,3],
                                 stride=[1,1],
                                 padding='VALID',
                                 biases_initializer = None)
        
        self.conv4 = slim.conv2d(inputs=self.conv3,
                                 num_outputs=h_size,
                                 kernel_size=[7,7],
                                 stride=[1,1],
                                 padding='VALID',
                                 biases_initializer = None)
        
        #We take the output from the final convolutional layer and split it into separated terms: advantage & value
        self.streamAC, self.streamVC = tf.split(self.conv4, 2, 3)
        self.streamA = slim.flatten(self.streamAC)
        self.streamV = slim.flatten(self.streamVC)
        
        xavier_init = tf.contrib.layers.xavier_initializer()
        
        self.AW = tf.Variable(xavier_init([h_size//2, env.actions]))
        self.VM = tf.Variable(xavier_init([h_size//2, 1]))
        
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VM)
        
        #Then combine to obtain the final Q-value //why substract mean from Advantage?
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage,axis=1, keep_dims=True))
        self.predict = tf.argmax(self.Qout, 1)
        
        #Obtain the loss by taking the sum of squares difference between the target and predicted Q values
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, env.actions, dtype=tf.float32) #// env.actions is probably number of actions
        
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)
        
        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)

# --- Experience replay implementation ---
class experience_buffer():
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size
        
    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)#// remove the necesary entries to make place for new experience
        
    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)),[size,5]) # buffer is experience tuple (size 4) plus "done" signal 

#Helper functions
def processState(states):
    return np.reshape(states,[84*84*3])

def updateTargetGraph(tfVars, tau):#// tau is a value smaller than 1 that limits the update value 
    total_vars = len(tfVars)
    op_holder = []
    
    for idx,var in enumerate(tfVars[0:total_vars//2]): #// select the first half of the variables (the one that belong to the action choosing net) 
        op_holder.append(
            tfVars[idx+total_vars//2].assign(
                (var.value()*tau)+((1-tau)*tfVars[idx+total_vars//2].value())))#//wtf?
    
    return op_holder

def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)           

# --- Training ---
#Set all the training parameters
batch_size = 32 #Number of experiences to use for each training step
update_freq = 4 #Frequency of a train step
y = .99 #Discount factor on the target Q-values
startE = 1 #Starting chance of random action
endE = 0.1 #Final chance of random action
anneling_steps = 10000. #Number of training steps to decrease the explorarion parameter e 
num_episodes = 10000 #Number of episodes of the game environmet on which to train the network
pre_train_steps = 10000 #Number of random action steps before training begins
max_epLength = 50 #Maximun allowed length of a game episode
load_model = False #Wheter to load a saved model
path = "./dqn_logs" #Directory for checkpoints
log_path = "./dqn_logs/logs" #Directory for logs
h_size = 512 #Number of activations of final conv layer
tau = 0.001 #Rate of update of target network values onto the primary network

f = open(log_path,'w')

tf.reset_default_graph()

# Network instances
mainQN = Qnetwork(h_size)
targetQN = Qnetwork(h_size)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

trainables = tf.trainable_variables()
targetOps = updateTargetGraph(trainables, tau)

myBuffer = experience_buffer()

#Set the rate of random action decrease
e = startE
stepDrop = (startE - endE)/anneling_steps

#Create lists to contain total reward and steps per episode
jList = []
rList = []
total_steps = 0

#Create path for checkpoints
if not os.path.exists(path):
    os.makedirs(path)

with tf.Session() as sess:
    sess.run(init)
    
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    
    #Set the target network to be equal to primary network
    updateTarget(targetOps, sess) 
    
    for i in range(num_episodes):
        episodeBuffer = experience_buffer()
        #Reset environment and get the first new observation
        s = env.reset()
        s = processState(s)
        d = False
        rAll = 0
        j = 0
        
        #Q-Network
        while j < max_epLength: #If the agent takes longer than 200 moves to reach either of the blocks, end the trail
            j+=1
            
            #Choose an action by greedily picking (with e chance of chosing randomly) from the Q-net
            if np.random.rand(1) < e or total_steps < pre_train_steps:
                a = np.random.randint(0,4)
            else: 
                a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput:[s]})[0]# //what is this extra [0]?
                
            s1, r, d = env.step(a)
            s1 = processState(s1)#// flatten the return frame
            total_steps += 1
            episodeBuffer.add(np.reshape(np.array([s,a,r,s1,d]),[1,5])) #Save the experience to the episode buffer
            
            if total_steps > pre_train_steps:
                if e > endE:
                    e -= stepDrop
                
                if total_steps % (update_freq) == 0:
                    trainBatch = myBuffer.sample(batch_size)
                    
                    #Perform the Double-DQN update to the target Q-values
                    Q1 = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,3])})#// why only pass the 3 col?
                    
                    Q2 = sess.run(targetQN.Qout, feed_dict={targetQN.scalarInput:np.vstack(trainBatch[:,3])})
                    end_multiplier = -(trainBatch[:,4]-1)
                    
                    doubleQ = Q2[range(batch_size), Q1]
                    targetQ = trainBatch[:,2] + (y * doubleQ * end_multiplier)
                    
                    #Update the network with our target values
                    _ = sess.run(mainQN.updateModel, \
                                feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,0]), mainQN.targetQ:targetQ,
                                          mainQN.actions:trainBatch[:,1]})
                    
                    #Set the target network to be equal to the primary network
                    updateTarget(targetOps, sess)
                
            rAll += r
            s = s1
                
            if d == True:
                break
                        
        myBuffer.add(episodeBuffer.buffer)
        jList.append(j)
        rList.append(rAll)
            
        #Periodically save the model
        if i % 1000 == 0:
            saver.save(sess, path + '/model-' + str(i) + '.cptk')
            print("Model Saved")
        if len(rList) % 10 == 0:
            accumR = np.mean(rList[-10:])
            log = str(total_steps) +"\t"+ str(accumR) +"\t"+ str(e) +"\n"
            print(log)
            f.write(log)
           
                
    saver.save(sess, path + '/model-' + str(i) + '.cptk')#// save final model?

final_log = "Percent of sucessful episodes: " + str(sum(rList)/num_episodes) + "%"
print(final_log)
f.write(final_log)
f.close()
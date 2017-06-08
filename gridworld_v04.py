"""
gridworld has been modified to return a simple array with 
the coordinates of the actor, the obstacles and the goal
"""

import numpy as np
import random 
import itertools
import scipy.misc
import matplotlib.pyplot as plt

class gameOb():
    def __init__(self, coordinates, size, intensity, channel, reward, name):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.intensity = intensity 
        self.channel = channel
        self.reward = reward
        self.name = name

class gameEnv():
    
    def __init__(self, partial, size):
        self.sizeX = size
        self.sizeY = size
        self.actions = 4 
        self.objects = []
        self.object_count = 0
        self.partial = partial
        
        #self.repetitionCheck = 0
        #self.flag = 0
        #self.pre1 = [-1, -1]
        #self.pre2 = [-1, -1]
        
        a = self.reset()
        #plt.imshow(a, interpolation = "nearest")
        
    def reset(self):
        self.objects = []
        
        hero = gameOb(self.newPosition(),1,1,2,None,'hero')
        self.objects.append(hero)
        
        #bug = gameOb(self.newFixedPosition(),1,1,1,4,'goal')
        bug = gameOb([10,11],1,1,1,4,'goal')
        self.objects.append(bug)
          
        #hole = gameOb(self.newFixedPosition(),1,1,0,-1,'fire')
        hole = gameOb([3,4],1,1,0,-1,'fire')
        self.objects.append(hole)
        
        #hole2 = gameOb(self.newFixedPosition(),1,1,0,-1,'fire')
        hole2 = gameOb([3,5],1,1,0,-1,'fire')
        self.objects.append(hole2)
        
        #hole3 = gameOb(self.newFixedPosition(),1,1,0,-1,'fire')
        hole3 = gameOb([3,6],1,1,0,-1,'fire')
        self.objects.append(hole3)
        
        #hole4 = gameOb(self.newFixedPosition(),1,1,0,-1,'fire')
        hole4 = gameOb([7,6],1,1,0,-1,'fire')
        self.objects.append(hole4)
        
        hole5 = gameOb([2,10],1,1,0,-1,'fire')
        self.objects.append(hole5)
        
        hole6 = gameOb([12,3],1,1,0,-1,'fire')
        self.objects.append(hole6)
        
        hole7 = gameOb([12,4],1,1,0,-1,'fire')
        self.objects.append(hole7)
        
        hole8 = gameOb([12,12],1,1,0,-1,'fire')
        self.objects.append(hole8)
        
        hole9 = gameOb([10,11],1,1,0,-1,'fire')
        self.objects.append(hole9)
        
        hole10 = gameOb([9,7],1,1,0,-1,'fire')
        self.objects.append(hole10)
        
        hole11 = gameOb([1,9],1,1,0,-1,'fire')
        self.objects.append(hole11)
        
        hole12 = gameOb([2,4],1,1,0,-1,'fire')
        self.objects.append(hole12)
        
        state = self.mapEnv()
        self.state = state
        
        #self.repetitionCheck = 0
        #self.flag = 0
        #self.pre1 = [-1, -1]
        #self.pre2 = [-1, -1]
        
        return state
    
    def reset_for_testing(self):
        self.objects = []
        
        hero = gameOb(self.newPosition(),1,1,2,None,'hero')
        self.objects.append(hero)
        
        #bug = gameOb(self.newFixedPosition(),1,1,1,4,'goal')
        bug = gameOb([10,11],1,1,1,4,'goal')
        self.objects.append(bug)
          
        #hole = gameOb(self.newFixedPosition(),1,1,0,-1,'fire')
        hole = gameOb([3,4],1,1,0,-1,'fire')
        self.objects.append(hole)
        
        #hole2 = gameOb(self.newFixedPosition(),1,1,0,-1,'fire')
        hole2 = gameOb([3,5],1,1,0,-1,'fire')
        self.objects.append(hole2)
        
        #hole3 = gameOb(self.newFixedPosition(),1,1,0,-1,'fire')
        hole3 = gameOb([3,6],1,1,0,-1,'fire')
        self.objects.append(hole3)
        
        #hole4 = gameOb(self.newFixedPosition(),1,1,0,-1,'fire')
        hole4 = gameOb([7,6],1,1,0,-1,'fire')
        self.objects.append(hole4)
        
        hole5 = gameOb([2,10],1,1,0,-1,'fire')
        self.objects.append(hole5)
        
        hole6 = gameOb([12,3],1,1,0,-1,'fire')
        self.objects.append(hole6)
        
        hole7 = gameOb([12,4],1,1,0,-1,'fire')
        self.objects.append(hole7)
        
        hole8 = gameOb([12,12],1,1,0,-1,'fire')
        self.objects.append(hole8)
        
        hole9 = gameOb([10,11],1,1,0,-1,'fire')
        self.objects.append(hole9)
        
        hole10 = gameOb([9,7],1,1,0,-1,'fire')
        self.objects.append(hole10)
        
        hole11 = gameOb([1,9],1,1,0,-1,'fire')
        self.objects.append(hole11)
        
        hole12 = gameOb([2,4],1,1,0,-1,'fire')
        self.objects.append(hole12)
        
        
        
        state = self.mapEnv()
        self.state = state
        
        #self.repetitionCheck = 0
        #self.pre1 = [(None, None)]
        #self.pre2 = [(None, None)]
        
        return state

    # add the 'if objectA.name != "hero":' 
    # To test when a hero is in a random position
    def newPosition(self):
        iterables = [range(self.sizeX), range(self.sizeY)]
        points = []
        
        for t in itertools.product(*iterables):
            points.append(t)
        
        currentPositions = []
        
        for objectA in self.objects:
            if objectA.name != "hero":
                if (objectA.x, objectA.y) not in currentPositions:
                    currentPositions.append((objectA.x, objectA.y))

        for pos in currentPositions:
            points.remove(pos)
            
        location = np.random.choice(range(len(points)),replace=False)
        return points[location]

    def newFixedPosition(self):
        
        self.object_count += 1
        iterables = [range(self.sizeX), range(self.sizeY)]
        points = []
        
        for t in itertools.product(*iterables):
            points.append(t)
        
        if self.object_count - 1 == 0:
            return points[0] 
        
        if self.object_count - 1 == 1:
            return points[-1] 
        
        points = points[self.sizeX:-self.sizeY]
        location = 6 * self.object_count % len(points) 
               
        return points[location]
        
    def mapEnv(self):
        hero = None
        a = []
        
        for item in self.objects:
            a.append(item.x)
            a.append(item.y)
            
        return a
    
    def renderEnv(self):
        #a = np.zeros([self.sizeY,self.sizeX,3])
        a = np.ones([self.sizeY+2, self.sizeX+2,3])
        a[1:-1, 1:-1, :] = 0
        
        hero = None
        
        for item in self.objects:
             
            a[item.y+1:item.y+item.size+1, item.x+1:item.x+item.size+1, item.channel] = item.intensity
            
            if item.name == 'hero':
                hero = item
                
        if self.partial == True:
            a = a[hero.y:hero.y+3,hero.x:hero.x+3,:]
        
        b = scipy.misc.imresize(a[:,:,0],[84,84,1],interp='nearest')
        c = scipy.misc.imresize(a[:,:,1],[84,84,1],interp='nearest')
        d = scipy.misc.imresize(a[:,:,2],[84,84,1],interp='nearest')
        a = np.stack([b,c,d],axis=2)
        
        plt.imshow(a, interpolation = "nearest")        
        
        return a

    def step(self, action):
        
        penalty = self.moveChar(action)
        reward, done = self.checkGoal()
        #state = self.renderEnv()
        state = self.mapEnv()        

        if reward == None:
            print(done)
            print(reward)
            print(penalty)

            return state, (reward+penalty), done
        else:
            return state, (reward+penalty), done
       
    # Call 'self.checkGoal_for_testing()' to testing
    def step_for_testing(self, action):
        
        penalty = self.moveChar(action)
        reward, done = self.checkGoal_for_testing()
        #state = self.renderEnv()
        state = self.mapEnv()
        
        if reward == None:
            print(done)
            print(reward)
            print(penalty)
            
            return state, (reward+penalty), done
        else:
            return state, (reward+penalty), done
        
                        
    def moveChar(self, direction):
        # 0 - up, 1 - down, 2 - left, 3 - right
        hero = self.objects[0]
        heroX = hero.x
        heroY = hero.y
        penalize = 0. 
        
        if direction == 0 and hero.y >= 1:
            hero.y -= 1
        if direction == 1 and hero.y <= self.sizeY-2:
            hero.y += 1
        if direction == 2 and hero.x >= 1:
            hero.x -= 1
        if direction == 3 and hero.x <= self.sizeX-2:
            hero.x += 1     
        if hero.x == heroX and hero.y == heroY:
            penalize = -1
        
        self.objects[0] = hero
        
        return penalize
    """
    A little modification 
        > if other.name == 'fire':
        >            self.objects[0] = gameOb((0,0),1,1,2,None,'hero')
    
    To make training speed faster, if the hero falls into the hole, just continue
    It refer to website 'http://computingkoreanlab.com/app/jAI/jQLearning/'
    This is a Q-learning test using Q-table. Looking at the options, they use penalty 'go on'(another is game over)
    I think that 'go on' option is faster more than 'game over'. So, I modified the code 
    """
    def checkGoal(self):
        others = []
        
        for obj in self.objects:
            if obj.name == "hero":
                hero = obj
            else:
                others.append(obj)
        
        ended = False
        
        for other in others:
            if hero.x == other.x and hero.y == other.y:
                #self.objects.remove(other)
                if other.name == 'fire':
                    self.objects[0] = gameOb((0,0),1,1,2,None,'hero')
                    return other.reward, False
                if other.name == 'goal':
                    self.objects[0] = gameOb((0,0),1,1,2,None,'hero')
                    return other.reward, True
            
        if ended == False:
            return 0.0, False  
    """
    This code is to testing 
    So, If hero falls into the hole or reach the goal, game is over 
    And heor's position changes per episode 
    """
    def checkGoal_for_testing(self):
        others = []
        
        for obj in self.objects:
            if obj.name == "hero":
                hero = obj
            else:
                others.append(obj)
        
        ended = False
        
        for other in others:
            if hero.x == other.x and hero.y == other.y:
                #self.objects.remove(other)
                self.objects[0] = gameOb(self.newPosition(),1,1,2,None,'hero')
                
                return other.reward, True
            
        if ended == False:
            return 0.0, False  
    """    
    def checkRepetition(self):
        if self.pre1[0] == -1 and self.pre1[1] == -1:
            self.pre1[0] = self.objects[0].x
            self.pre1[1] = self.objects[0].y
            return False
        
        if self.pre2[0] == -1 and self.pre2[1] == -1:
            self.pre1[0] = self.objects[0].x
            self.pre1[1] = self.objects[0].y
            return False
        
        if self.flag == 0 and self.pre1[0] != self.objects[0].x and self.pre1[1] != self.objects[0].y:
            self.pre1[0] = self.objects[0].x
            self.pre1[1] = self.objects[0].y
            self.flag = 1
            return False    
        
        if self.flag == 1 and self.pre2[0] != self.objects[0].x and self.pre2[1] != self.objects[0].y:
            self.pre2[0] = self.objects[0].x
            self.pre2[1] = self.objects[0].y
            self.flag = 0
            return False  
               
        if self.pre1[0] == self.objects[0].x and self.pre1[1] == self.objects[0].y:
            self.repetitionCheck += 1
            return False    
        
        if self.pre2[0] == self.objects[0].x and self.pre2[1] == self.objects[0].y:
            self.repetitionCheck += 1
            return False    
        
        if self.repetitionCheck >= 5:
            return True
    """
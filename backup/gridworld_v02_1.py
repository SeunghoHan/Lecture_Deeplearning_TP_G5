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
        #selef.action = 4
        self.actions = 4
        self.objects = []
        self.positions = [(0,0), (1,2), (2,1), (1,4), (3,3), (4,4)]
        self.partial = partial
        a = self.reset()
        plt.imshow(a, interpolation = "nearest")

    def reset(self):
        self.objects = []

        hero = gameOb(self.positions[0],1,1,2,None,'hero')
        self.objects.append(hero)

        bug = gameOb(self.positions[5],1,1,1,1,'goal')
        self.objects.append(bug)

        #bug2 = gameOb(self.newPosition(),1,1,1,1,'goal')
        #self.objects.append(bug2)

        #bug3 = gameOb(self.newPosition(),1,1,1,1,'goal')
        #self.objects.append(bug3)

        #bug4 = gameOb(self.newPosition(),1,1,1,1,'goal')
        #self.objects.append(bug4)

        hole = gameOb(self.positions[1],1,1,0,-1,'fire')
        self.objects.append(hole)

        hole2 = gameOb(self.positions[2],1,1,0,-1,'fire')
        self.objects.append(hole2)

        hole3 = gameOb(self.positions[3],1,1,0,-1,'fire')
        self.objects.append(hole3)

        hole4 = gameOb(self.positions[4],1,1,0,-1,'fire')
        self.objects.append(hole4)

        state = self.renderEnv()
        self.state = state

        return state

    def newPosition(self):
        iterables = [range(self.sizeX), range(self.sizeY)]
        points = []

        # create coordinates (x, y) within range (size, size)
        for t in itertools.product(*iterables):
            points.append(t)

        currentPositions = []

        # check obejects assigned coordinates
        for objectA in self.objects:
            if (objectA.x, objectA.y) not in currentPositions:
                currentPositions.append((objectA.x, objectA.y))

        for pos in currentPositions:
            points.remove(pos)

        # rnadomly assign coordinate to obejct that has no coordinates  
        location = np.random.choice(range(len(points)),replace=False)
        return points[location]

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

        return a

    def step(self, action):

        penalty = self.moveChar(action)
        reward, done = self.checkGoal()
        state = self.renderEnv()

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
            penalize = 0.0

        self.objects[0] = hero

        return penalize

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
                self.objects[0] = gameOb(self.newPosition(),1,1,2,None,'hero')

                return other.reward, True

        if ended == False:
            return 0.0, False  

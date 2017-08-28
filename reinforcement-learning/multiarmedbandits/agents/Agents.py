import math
import random

import numpy as np


# Interface for an agent
class Agent:
    bestArm = 1

    def __init__(self, bandit, numberOfArms):
        self.bandit = bandit
        self.numberOfArms = bandit.numberOfArms
        self.actionValueEstimates = np.zeros(self.numberOfArms)

    def chooseArm(self):
        raise NotImplementedError('Not implemented yet')

    def getReward(self, armChosen):
        return self.bandit.getReward(armChosen)

    def updateEstimate(self, arm, reward):
        raise NotImplementedError('Not implemented yet')

    def learn(self):
        armChosen = self.chooseArm()
        reward = self.getReward(armChosen)
        self.updateEstimate(armChosen, reward)


class AgentEpsilonGreedy(Agent):
    def __init__(self, bandit, numberOfArms, epsilon):
        Agent.__init__(self, bandit, numberOfArms)
        self.epsilon = epsilon
        self.armSelectionCount = np.zeros(numberOfArms)

    def getAlpha(self, arm):
        return 1.0 / self.armSelectionCount[arm - 1]

    def updateEstimate(self, arm, reward):
        self.actionValueEstimates[arm - 1] \
            += self.getAlpha(arm) * (reward - self.actionValueEstimates[arm - 1])
        if (self.actionValueEstimates[arm - 1]
                > self.actionValueEstimates[self.bestArm - 1]):
            self.bestArm = arm

    def shouldExplore(self):
        return random.random() <= self.epsilon

    def chooseArm(self):
        chosenArm = None
        if self.shouldExplore():
            chosenArm = random.randint(1, self.numberOfArms)
        else:
            chosenArm = self.bestArm
        self.armSelectionCount[chosenArm - 1] += 1
        return chosenArm


class AgentSoftmax(Agent):
    def __init__(self, bandit, numberOfArms, epsilon, temperature):
        Agent.__init__(self, bandit, numberOfArms)
        self.epsilon = epsilon
        self.temperature = temperature
        self.armSelectionCount = np.zeros(numberOfArms)

    def getAlpha(self, arm):
        return 1.0 / self.armSelectionCount[arm - 1]

    def weightedChoice(self, choices):
        total = sum(choices)
        r = random.uniform(0, total)
        upto = 0
        for c, w in zip(range(1, len(choices) + 1), choices):
            if upto + w >= r:
                return c
            upto += w

    def chooseArm(self):
        choices = [math.exp(q / self.temperature) for q in self.actionValueEstimates]
        totalSum = sum(choices)
        choices = [i / totalSum for i in choices]
        chosenArm = self.weightedChoice(choices)
        self.armSelectionCount[chosenArm - 1] += 1
        return chosenArm

    def updateEstimate(self, arm, reward):
        self.actionValueEstimates[arm - 1] \
            += self.getAlpha(arm) * (reward - self.actionValueEstimates[arm - 1])
        if (self.actionValueEstimates[arm - 1]
                > self.actionValueEstimates[self.bestArm - 1]):
            self.bestArm = arm

from agents.Agents import AgentEpsilonGreedy, AgentSoftmax
from bandit.Bandit import MultiArmedBandit
from commons import generateActionValues


class TestBed:
    def __init__(self, bandits, agents):
        self.numberOfBandits = len(bandits)
        self.bandits = bandits
        self.agents = agents

    def learn(self):
        for agent in self.agents:
            agent.learn()

    def getAverageReward(self):
        reward = 0
        for agent in self.agents:
            reward += agent.getReward(agent.bestArm)
        return reward / self.numberOfBandits


class EpsilonGreedyTestBed(TestBed):
    def __init__(self, numberOfBandits, epsilon, numberOfArms, armClass):
        bandits = [MultiArmedBandit(armClass, numberOfArms, generateActionValues(numberOfArms)) for _ in
                   range(0, numberOfBandits)]
        agents = [AgentEpsilonGreedy(bandits[i], numberOfArms, epsilon) for i in range(0, numberOfBandits)]
        TestBed.__init__(self, bandits, agents)


class SoftmaxTestBed(TestBed):
    def __init__(self, numberOfBandits, epsilon, temperature, numberOfArms, armClass):
        bandits = [MultiArmedBandit(armClass, numberOfArms, generateActionValues(numberOfArms)) for _ in
                   range(0, numberOfBandits)]
        agents = [AgentSoftmax(bandits[i], numberOfArms, epsilon, temperature) for i in range(0, numberOfBandits)]
        TestBed.__init__(self, bandits, agents)

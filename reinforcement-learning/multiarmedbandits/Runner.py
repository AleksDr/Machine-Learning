import matplotlib.pyplot as plt

from TestBeds import EpsilonGreedyTestBed
from TestBeds import SoftmaxTestBed
from bandit.Arms import GaussianArm

# Some really ugly code for graphs building and testing of the agents
if __name__ == '__main__':
    numberOfBandits = 2000
    numberOfArms = 10
    numberOfIterations = 2000
    epsilonFirst = 0.0
    epsilonSecond = 0.01
    epsilonThird = 0.1
    temperatureFirst = 1
    temperatureSecond = 100
    temperatureThird = 100000

    testBedSoftFirst = SoftmaxTestBed(numberOfBandits, epsilonThird, temperatureFirst, numberOfArms, GaussianArm)
    testBedSoftSecond = SoftmaxTestBed(numberOfBandits, epsilonThird, temperatureSecond, numberOfArms, GaussianArm)
    testBedSoftThird = SoftmaxTestBed(numberOfBandits, epsilonThird, temperatureThird, numberOfArms, GaussianArm)
    testBedGreedyFirst = EpsilonGreedyTestBed(numberOfBandits, epsilonFirst, numberOfArms, GaussianArm)
    testBedGreedySecond = EpsilonGreedyTestBed(numberOfBandits, epsilonSecond, numberOfArms, GaussianArm)
    testBedGreedyThird = EpsilonGreedyTestBed(numberOfBandits, epsilonThird, numberOfArms, GaussianArm)

    rewardsSoftFirst = []
    rewardsSoftSecond = []
    rewardsSoftThird = []
    rewardsGreedyFirst = []
    rewardsGreedySecond = []
    rewardsGreedyThird = []

    # Running the learning process
    for i in range(0, numberOfIterations):
        testBedSoftFirst.learn()
        rewardsSoftFirst.append(testBedSoftFirst.getAverageReward())
        testBedSoftSecond.learn()
        rewardsSoftSecond.append(testBedSoftSecond.getAverageReward())
        testBedSoftThird.learn()
        rewardsSoftThird.append(testBedSoftThird.getAverageReward())

        testBedGreedyFirst.learn()
        rewardsGreedyFirst.append(testBedGreedyFirst.getAverageReward())
        testBedGreedySecond.learn()
        rewardsGreedySecond.append(testBedGreedySecond.getAverageReward())
        testBedGreedyThird.learn()
        rewardsGreedyThird.append(testBedGreedyThird.getAverageReward())

    # Different epsilon parameters for greedy epsylon agent
    plt.ylabel('average reward')
    plt.xlabel('iteration')
    plt.plot(rewardsGreedyFirst, label='Greedy, e=0')
    plt.plot(rewardsGreedySecond, label='Greedy, e=0.01')
    plt.plot(rewardsGreedyThird, label='Greedy, e=0,1')
    plt.title('Epsilon greedy with different epsilons')
    plt.legend()
    plt.savefig('graphs/greedy_comprasion.png')
    plt.clf()
    plt.cla()
    plt.close()

    # Different temperature parameters for softmax agent
    plt.ylabel('average reward')
    plt.xlabel('iteration')
    plt.plot(rewardsSoftFirst, label='Softmax, t=1, e=0.1')
    plt.plot(rewardsSoftSecond, label='Softmax, t=100, e=0.1')
    plt.plot(rewardsSoftThird, label='Softmax, t=10000, e=0.1')
    plt.title('Softmax with different temperatures')
    plt.legend()
    plt.savefig('graphs/softmax_comprasion.png')
    plt.clf()
    plt.cla()
    plt.close()

    # Comprasion of all agents
    plt.ylabel('average reward')
    plt.xlabel('iteration')
    plt.plot(rewardsGreedyThird, label='Greedy, e=0,1')
    plt.plot(rewardsSoftThird, label='Softmax, t=10000, e=0.1')
    plt.title('Epsilon greedy vs softmax')
    plt.legend()
    plt.savefig('graphs/greedyVSsoftmax_comprasion.png')
    plt.clf()
    plt.cla()
    plt.close()

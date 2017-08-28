class MultiArmedBandit:
    def __init__(self, armClass, numberOfArms, actionValueArray):
        self.numberOfArms = numberOfArms
        self.arms = [armClass(actionValueArray[i]) for i in range(0, numberOfArms)]

    def getReward(self, armPulled):
        return self.arms[armPulled - 1].draw()

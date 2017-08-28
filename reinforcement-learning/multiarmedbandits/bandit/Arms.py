from numpy import random


# Interface for an arm
class Arm:
    def draw(self):
        raise NotImplementedError("Call this method from an instance of a subclass.")


class GaussianArm(Arm):
    def __init__(self, actionValue):
        self.actionValue = actionValue

    def draw(self):
        return self.actionValue + random.normal()

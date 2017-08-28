from numpy import random


# TODO - need more ways of initializing for the action values (look at 'optimistic initial values')
def generateActionValues(numberOfArms):
    return random.normal(0, 1, numberOfArms)

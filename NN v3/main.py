# ================================================================
# =           main script for init of neural net class           =
# ================================================================

import random
import numpy as np


from NeuralNetwork import NeuralNetwork

t = [(0,0,0), (1,0,1), (0,1,1), (1,1,0)]
inputNodes = 2
hiddenNodes = [5]
outputNode = 1
learningRate = 0.1
trainingCycles = 20000

nn = NeuralNetwork(inputNodes, hiddenNodes, outputNode, learningRate)


for x in range(trainingCycles):
    r = random.randint(0,3)
    inputs = [t[r][0], t[r][1]]
    targets = [t[r][2]]
    nn.train(inputs, targets)

    percent = str(int(x/trainingCycles * 100)) + "%"
    print(percent)



# nn.loadState('xorState')

print(nn.guess([0,0]))
print(nn.guess([1,0]))
print(nn.guess([0,1]))
print(nn.guess([1,1]))


# nn.saveState('xorState')



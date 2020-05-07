import sys
import numpy as np
from os import path
import emnist


from PIL import Image

path = sys.path[0]

sys.path.append('NN v3')
sys.path.append('%s/imageRecognition' % (path))

from NeuralNetwork import NeuralNetwork
import irs


filename = "letterClassiferModel400"
foldername = "weights and labels data"
doSave = False
doTest = False


def norm(data):
    newList = []
    for x in range(len(data)):
        y = data[x]
        y = y / 255
        newList.append(y)
    return newList


def toTargets(label):
    targets = [0] * 26
    targets[label - 1] = 1
    return targets

def whatIndex(g):
    return np.argmax(g)

def indexToChar(index):
    return chr(index + 97)

def test():

    testData, testLabels = emnist.extract_test_samples('letters')
    # import testing data from emnist
    # call them data and labels


    correct = 0
    for x in range(testData.shape[0]):
        inputs = norm(np.ndarray.flatten(testData[x]))
        guess = nn.guess(inputs)
        print("the number was: " + str(testLabels[x]))
        guess = whatIndex(guess) + 1
        print("it guessed it was :" + str(guess))
        if guess == testLabels[x]:
            correct += 1

    # print("it correctly predicted " + str(correct / len(data) * 100) + "%")
    return (correct / testData.shape[0] * 100)



nn = NeuralNetwork(784,[400],26, 0.1)

if doSave:

    trainingData, trainingLabels = emnist.extract_training_samples('letters')

    trained = False
    bestResult = 0
    # remember to randomise the 2d array when going over another epoch!!!!!!!!!
    while not trained:
        oldPercent = 0
        for x in range(trainingData.shape[0]):
            inputs = norm(np.ndarray.flatten(trainingData[x]))
            targets = toTargets(trainingLabels[x])
            percent = int(x/trainingData.shape[0] * 100)
            if percent > oldPercent:
                oldPercent = percent
                print(percent)
            nn.train(inputs, targets)

        newResult = test()
        print("This epoch was %s the previous was %s" % (newResult, bestResult))
        if newResult > bestResult:
            bestResult = newResult
        else:
            trained = True


    nn.saveState(filename)


nn.loadState(filename)

if doTest:
    test()


print('working...')
inputs = irs.handWrittenNumberData()

lettersArray = []
if inputs != None:
    for image in inputs:
        if image == "space":
            lettersArray.append(" ")
        elif image == "para":
            lettersArray.append('\n')
        else:
            normInput = norm(image)
            guess = nn.guess(normInput)
            indexValue = whatIndex(guess)
            lettersArray.append(indexToChar(indexValue))

with open("Finished Text", "w") as file:
    for letter in lettersArray:
        file.write(letter)




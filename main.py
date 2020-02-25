import gettingData
import sys
import numpy as np
from os import path


from PIL import Image

path = sys.path[0]

sys.path.append('../NN v3')
sys.path.append('%s/imageRecognition' % (path))
sys.path.append('../../email downloader')
from NeuralNetwork import NeuralNetwork
import irs


filename = "digitClassifierState"
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
    targets = [0] * 10
    targets[label] = 1
    return targets



def whatNumber(g):
    return np.argmax(g)


def test():

    images_filename = "%s/data/t10k-images.idx3-ubyte" % (path)
    labels_filename = "%s/data/t10k-labels.idx1-ubyte" % (path)
    data, labels = gettingData.getData(images_filename, labels_filename)


    # newPixels = []
    # counter = 0

    # for i in range(28):
    #     subList = []
    #     for j in range(28):
    #         pixelVal = data[23][counter]
    #         subList.append(pixelVal)
    #         counter += 1
    #     newPixels.append(subList)

    # array = np.array(newPixels, dtype=np.uint8)
    # formatted = Image.fromarray(array)
    # formatted.show()
    # print(labels[23])

    correct = 0
    for x in range(len(data)):
        inputs = norm(data[x])
        guess = nn.guess(inputs)
        print("the number was: " + str(labels[x]))
        guess = whatNumber(guess)
        print("it guessed it was :" + str(guess))
        if guess == labels[x]:
            correct += 1

    # print("it correctly predicted " + str(correct / len(data) * 100) + "%")
    return (correct / len(data) * 100)



nn = NeuralNetwork(784,[100,50],10, 0.1)

if doSave:

    images_filename = "%s/data/train-images.idx3-ubyte" % (path)
    labels_filename = "%s/data/train-labels.idx1-ubyte" % (path)
    data, labels = gettingData.getData(images_filename, labels_filename)

    trained = False
    bestResult = 0
    # remember to randomise the 2d array when going over another epoch!!!!!!!!!
    while not trained:
        for x in range(len(data)):
            inputs = norm(data[x])
            targets = toTargets(labels[x])
            print(int(x/len(data) * 100))
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

# images_filename = "%s/data/t10k-images.idx3-ubyte" % (path)
# labels_filename = "%s/data/t10k-labels.idx1-ubyte" % (path)
# data, labels = gettingData.getData(images_filename, labels_filename)


print('computing number... ')
inputs = norm(irs.handWrittenNumberData())

guess = nn.guess(inputs)
print(whatNumber(guess))



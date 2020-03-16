import numpy as np
import random as r
import pickle
import os
from itertools import chain

# ===============================================
# =           neural network library!           =
# ===============================================


class NeuralNetwork:

    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        self.inputNodes = inputNodes
        self.hiddenNodes = hiddenNodes
        self.outputNodes = outputNodes
        self.learningRate = learningRate
        self.initialInputs = []
        self.initialTargets = []

        self.nnLayerInfo = self.nnLayerInfoInit()
        self.iterationList = [len(self.nnLayerInfo) - 1, 0]

        self.weights = self.weightInitilisation()
        self.bias = self.biasInitilisation()
        self.newWeights = []
        self.newBias = []
        self.outputs = []

        self.weightsAndBiasFolderName = 'weights and labels data'

# ================================================
# =           initilisation functions!           =
# ================================================

    def nnLayerInfoInit(self):
        # creates a list of how many nodes are in each layer of
        # the neural network
        # returns a list

        l = []
        l.append(self.inputNodes)
        for n in self.hiddenNodes:
            l.append(n)
        l.append(self.outputNodes)
        return l

    def weightInitilisation(self):

        # initilsing the weights as a 2d list, each sublist is
        # representitive of how many weights are in each layer of the network
        # returns a 2d array

        weights = []
        weightsTemp = []
        n = []
        lenHidden = len(self.hiddenNodes)

        n.append([self.inputNodes * self.hiddenNodes[0]])

        for x in range(lenHidden - 1):
            n.append([self.hiddenNodes[x] * self.hiddenNodes[x+1]])

        n.append([self.hiddenNodes[lenHidden - 1] * self.outputNodes])
        # n is a 2d list of how many weights should be in
        # each layer of the network

        for x in range(len(n)):
            for y in range(n[x][0]):
                weightsTemp.append(2 * r.random() - 1)
                #creating the random numbers between -1 and 1 for the weights
            weights.append(weightsTemp)
            #storing the list of weights in to the weights list
            # to make it a 2d array
            weightsTemp = []

        return weights

    def biasInitilisation(self):
        # initilises the bias of each node not including the
        # first input nodes
        # returns a 2d array

        bias = []
        biasTemp = []

        for x in self.nnLayerInfo[1:]:
            for y in range(x):
                biasTemp.append(2 * r.random() - 1)

            bias.append(biasTemp)
            biasTemp = []

        return bias

# ======  End of initilisation functions!  =======

# =========================================
# =           utility functions           =
# =========================================

    def toMatrix(self, c):

        # takes an char as input and matches it with the relevent
        # switch statement. it then returns a matrix of specified
        # data set

        switch = {
            'w': [self.weights, (self.nnLayerInfo[self.iterationList[1] + 1],
                 self.nnLayerInfo[self.iterationList[1]])],
            'b': [self.bias, (self.nnLayerInfo[self.iterationList[1] + 1], 1)],
            'i': [[self.initialInputs], (len(self.initialInputs), 1)],
            't': [[self.initialTargets], (len(self.initialTargets), 1)],
            'o': [self.outputs, (self.nnLayerInfo[self.iterationList[1] + 1], 1)]
        }

        data = np.array(switch.get(c)[0][self.iterationList[1]])
        shape = switch.get(c)[1]
        matrix = data.reshape(shape)
        return matrix

    def sigmoid(self, x):
        #normalising function returns sigmoid value of
        # given input x which is a matrix
        return 1 / (1 + np.exp(- x))

    def derivSigmoid(self, x):
        # derivitive of the sigmoid function and returns
        # a matrix of given input x
        return x * (1 - x)

    def saveState(self, name):
        try:
           os.mkdir(self.weightsAndBiasFolderName)
        except FileExistsError:
            pass

        fileName = self.weightsAndBiasFolderName + '/' + name
        state = [self.weights, self.bias]
        with open(fileName, 'wb') as f:
            pickle.dump(state, f)


    def loadState(self, name):
        fileName = self.weightsAndBiasFolderName + '/' + name

        try:
            with open(fileName, 'rb') as f:
                state = pickle.load(f)
                self.weights = state[0]
                self.bias = state[1]
        except:
            print("No state has been saved yet")


# ======  End of utility functions  =======

# ======================================
# =           main functions           =
# ======================================

    def train(self, inputs, targets):
        # the main training function
        self.iterationList[1] = 0
        #converting inputs to a matrix
        self.initialInputs = inputs
        inputs = self.toMatrix('i')

        #converting targets to matrix
        self.initialTargets = targets
        targets = self.toMatrix('t')

        #generating an output from the feed forward function
        output = self.feedForward(inputs)

        #calculating the output error for the last node(s)
        #in the neural net
        outputErrors = targets - output

        #back propergate through the neural network
        self.backProp(outputErrors)

        #change all of the old weights and bia to the new created weights
        #and bias from back propergation
        self.newWeights = self.newWeights[::-1]#reversing the list
        self.weights = self.newWeights

        self.newBias = self.newBias[::-1]
        self.bias = self.newBias

        #wiping all data from lists for use again
        self.newWeights = []
        self.newBias = []
        self.outputs = []



    def feedForward(self, inputs):
        # feeds outputs forward through each layer of the neural network
        weights = self.toMatrix('w')  # we want weights to be returned so
        # we use 'w' as the parameter
        bias = self.toMatrix('b')   # we want a bias to be returned so
        # we use 'b' as the parameter

        #equation for getting and output of a single layer
        output = self.sigmoid(np.dot(weights, inputs) + bias)

        #adds the outputs to a list to access later
        self.outputs.append(output.T.tolist()[0])

        #adds an iterations to the iteration list
        self.iterationList[1] += 1

        #escape logic, if the iteration == max iterations then
        # return the output
        if self.iterationList[1] == self.iterationList[0]:
            return output
        else: # else, keep iterating of this function recursively
            return self.feedForward(output) #this layers output is used
            # as the next iterations input

    def backProp(self, outputErrors):
        # back properagting through the layers of the
        # neural network to change the wieghts and bias

        # decrease the iteration list to evetually get to 0
        # due to use working backwards.
        self.iterationList[1] -= 1

        #grab the relevent weights from the weights list
        weights = self.toMatrix('w')
        #grab the relevent bias from the bias list
        bias = self.toMatrix('b')

        #calculating the input errors from section
        inputErrors = np.dot(weights.T, outputErrors)

        #brabbing the relevent output from the output list
        outputMatrix = self.toMatrix('o')

        #grabbing the inputs for the relevent section
        #unless its at the beginning of the neural network
        if self.iterationList[1] != 0:
            self.iterationList[1] -= 1
            inputMatrix = self.toMatrix('o')
            self.iterationList[1] += 1
        else: # uses the initialinput list instead
        # to stop index out of range error
            inputMatrix = self.toMatrix('i')

        #calculting the gradient
        gradient = self.derivSigmoid(outputMatrix)
        gradient = np.multiply(gradient, outputErrors)
        gradient = np.multiply(gradient, self.learningRate)

        # calculating the change in weights from gradient
        changeInWeights = np.dot(gradient, inputMatrix.T)
        #calculating the new weights and storing them for later
        #manipulation
        newWeights = changeInWeights + weights
        newWeights = newWeights.tolist()
        newWeights = list(chain.from_iterable(newWeights))
        self.newWeights.append(newWeights)

        #calculating the new bias and storing them for later
        #maninpulation
        newBias = np.add(bias, gradient)
        newBias = newBias.tolist()
        newBias = list(chain.from_iterable(newBias))
        self.newBias.append(newBias)

        #if we are not at the first section of the neural
        #network then keep back propergating...
        if self.iterationList[1] != 0:
            return self.backProp(inputErrors)

    def guess(self, inputs):
        # the user gives some data for the computer to guess
        # what value it is

        # resetting the iteration list to 0 because
        # we dont back propergate which would normally
        # reset the value
        self.iterationList[1] = 0

        #inputs to matrix..
        i = np.array(inputs)
        inputs = i.reshape(len(inputs), 1)

        #return guess
        return self.feedForward(inputs)

# ======  End of main functions  =======


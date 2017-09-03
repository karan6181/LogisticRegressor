__author__ = "Karan Jariwala"
# FileName: logreg


import random, math, sys
from matplotlib import pyplot as myplot

class FileOperation:
    """
    This class reads the data from the csv file and stores in a data
    structure. The data in a file has to be in proper format e.g.: first two
    column is an input value and the third column is an output classifier
    value( Total three columns separated by commas ). It also writes the
    data into the csv file in the same format as above.
    """
    __slots__ = "inputListForOutput1", "inputListForOutput0", \
                "combinedInputList", "outputList", "fileName", "weight"

    def __init__(self, fileName=None):
        """
        This initializes the data structure and store the input filename.

        :param fileName: Input csv file name
        """
        if not fileName:
            print("ERROR!!! Please specify a csv file name")
            sys.exit(1)
        else:
            self.fileName = fileName
        self.inputListForOutput1 = list()
        self.inputListForOutput0 = list()
        self.combinedInputList = list()
        self.outputList = list()
        self.weight = list()

    def readFile(self):
        """
        It reads the data from the file and store in a specific data structure.
        combinedInputList: It has three column and as many rows as in the input
                           file. Here the first column is always one because we
                           need to multiply this with bias.
                           Format is [ 1, inputValue1, inputValue2 ]
        inputListForOutput0:It has two column and as many rows as in the input
                           file. It is separated based on classifier output 0.
        inputListForOutput1:It has two column and as many rows as in the input
                           file. It is separated based on classifier output 1.
        outputList:        It has one column and as many rows as in the input
                           file. It reads the last column from the file which
                           classifier 0 or 1.
        :return: None
        """
        with open(self.fileName) as fp:
            for line in fp.readlines():
                line = line.strip().split(",")
                self.combinedInputList.append([1, float(line[0]), float(line[1])])
                self.outputList.append( float(line[2]) )
                if float(line[2]) == 0.0:
                    self.inputListForOutput0.append([float(line[0]), float(line[1])])
                elif float(line[2]) == 1.0:
                    self.inputListForOutput1.append([float(line[0]), float(line[1])])

    def writeFile(self, weightList):
        """
        It writes the weights data into the file which is 3 column wide and
        as many rows as in the input file.

        :param weightList: A list of weights
        :return: None
        """
        with open("weights.csv", "w+") as fp:
            for weight in weightList:
                wtString = str(weight[0]) + "," + str(weight[1]) + \
                           "," + str(weight[2]) + "\n"
                fp.write(wtString)

    def readWeightFile(self):
        """
        It read the weights.csv file and store the last weight

        :return: None
        """
        with open("weights.csv", "r") as fp:
            for line in fp.readlines():
                line = line.strip().split(",")
                self.weight = [float(line[0]), float(line[1]), float(line[2])]

class NeuronUnit:
    """
    A Neuron networks class consist of weights and synopsis and sigmod
    activation function.
    """
    __slots__ = 'weight'

    def __init__(self, weight=list()):
        """
        It initializes the weights randomly

        :param weight: A list of floats with 1x3 dimension
        """
        if not weight:
            self.weight = [random.random(), random.random(), random.random()]
        else:
            self.weight = weight

    def activation(self, inputList):
        """
        It multiplies the input value with weights and apply the sigmoid
        activation function.

        :param inputList: A list of input
        :return: A sigmoid activation value
        """
        synopsis = self.matrixMul(inputList, self.weight)
        return self.sigmoid(synopsis)

    def matrixMul(self, list1, list2):
        """
        It multiplies the two matrix using dot product.

        :param list1: A list of size 1x3 dimension
        :param list2: A list of size 1x3 dimension
        :return: final dot product value
        """
        val = 0
        for i in range(len(list1)):
            val += list1[i]*list2[i]
        return val

    def sigmoid(self, acti):
        """
        A sigmoid activation function.

        :param acti: A final dot product value
        :return: a sigmoid value
        """
        return 1/(1 + math.exp((-1)*acti))

    def getWeights(self):
        """
        A getter method to return the weights

        :return: return the weight
        """
        return self.weight

    def setWeights(self, weight):
        """
        A setter method to set the weights

        :param weight: A list of weights
        :return: None
        """
        self.weight = weight

    def __str__(self):
        """
        Overrides the string method to return the input weights and a bias

        :return: return the input weights and a bias
        """
        return "Weights: %s,\tBias: %s" %( self.weight[1:], self.weight[0] )

class LogisticRegression:
    """
    A Logistic regression class which trained the network using batch
    gradient descent.
    It also plots the Sum of Square Error graph and decision boundary graph.
    """
    __slots__ = "neuronObj", "fileObject", "SSERecords", \
                "epochs", "learningRate"

    def __init__(self, neuronObj=None, fileObject=None):
        """
        It initializes the neuron class object and file operation class object
        and also the number of epochs and a learning rate.

        :param neuronObj: A neuron class object
        :param fileObject: A file operation class object
        """
        if not neuronObj or not fileObject:
            print("Please specify Neuron object and file object")
            sys.exit(1)
        else:
            self.neuronObj = neuronObj
            self.fileObject = fileObject
            self.SSERecords = list()
            self.epochs = 1000
            self.learningRate = 0.1

    def batchGradientDescent(self):
        """
        It trained or optimized the networks using batch gradient descent and
        back propagation.
        It also stores the sum of square errors for every epoch.

        :return: A list of final optimized weights
        """
        combinedWeights = list()
        for _ in range(self.epochs):
            newWeight = self.neuronObj.getWeights()
            sumOfSquareErr = 0
            for dataRowIp, dataRowOp in zip(self.fileObject.combinedInputList,
                                            self.fileObject.outputList):
                activation = self.neuronObj.activation(dataRowIp)
                sumOfSquareErr += (dataRowOp - activation )**2
                delta = ( activation - dataRowOp ) * activation * ( 1 - activation )
                newWeight = [ wt - ( self.learningRate * delta * val )
                              for val, wt in zip(dataRowIp, newWeight) ]
            combinedWeights.append(newWeight)
            self.neuronObj.setWeights(newWeight)
            self.SSERecords.append(sumOfSquareErr/len(self.fileObject.combinedInputList))
        return combinedWeights

    def correctIncorrectSamples(self):
        """
        It calculates the class zero and class one correct and Incorrect
        samples.

        :return: None
        """
        classZeroCorrect = 0
        classZeroIncorrect = 0
        classOneCorrect = 0
        classOneIncorrect = 0

        inputList = self.fileObject.combinedInputList
        classifierList = self.fileObject.outputList
        for myList, cl in zip(inputList, classifierList ):
            val = self.neuronObj.activation(myList)
            classifier = self.findClassifier(val)
            if cl == 0:
                if cl == classifier:
                    classZeroCorrect += 1
                else:
                    classZeroIncorrect += 1
            elif cl == 1:
                if cl == classifier:
                    classOneCorrect += 1
                else:
                    classOneIncorrect += 1

        print("Number of correct class 0 sample:\t", classZeroCorrect)
        print("Number of incorrect class 0 sample:\t", classZeroIncorrect)
        print("Number of correct class 1 sample:\t", classOneCorrect)
        print("Number of incorrect class 0 sample:\t", classOneIncorrect)

    def findClassifier(self, val):
        """
        It finds the classifier it belongs to.

        :param val: A sigmod value
        :return: Either 0 or 1
        """
        if val < 0.5:
            return 0
        else:
            return 1

    def plotGraph(self, figure):
        """
        It plots the graph of number of epochs verses sum of square error
        for that epochs.

        :param figure: A matplotlib library object
        :return: None
        """
        epoch = list(range(self.epochs))
        ssePlot = figure.add_subplot(121)
        ssePlot.plot(epoch, self.SSERecords, label='Training')
        ssePlot.legend(loc='upper right')

        ssePlot.set_title("SSE vs Epoch")
        ssePlot.set_xlabel("EPOCH( pass over the entire training set )")
        ssePlot.set_ylabel("SSE( sum of squared error for all samples )")

    def decisionBoundaryPlot(self, figure):
        """
        It plots the decision boundary graph with class zero input values on
        x-axis and class one values on y-axis.

        :param figure: A matplotlib library object
        :return: None
        """
        x1Output0 = list()
        x1Output1 = list()
        x2Output0 = list()
        x2Output1 = list()

        for i in range(len(self.fileObject.inputListForOutput1)):
            x1Output0.append(self.fileObject.inputListForOutput1[i][0])
            x1Output1.append(self.fileObject.inputListForOutput1[i][1])

        for i in range(len(self.fileObject.inputListForOutput0)):
            x2Output0.append(self.fileObject.inputListForOutput0[i][0])
            x2Output1.append(self.fileObject.inputListForOutput0[i][1])

        decBoundPlot = figure.add_subplot(122)
        decBoundPlot.scatter(x1Output0, x1Output1, c='r', label="Class 0")
        decBoundPlot.scatter(x2Output0, x2Output1, c='g', label="Class 1")
        decBoundPlot.legend(loc='upper left')

        minMax = [[min(inputVal), max(inputVal)]
                  for inputVal in zip(*self.fileObject.combinedInputList)]
        minValue = min(minMax[1][0], minMax[2][0])
        maxValue = max(minMax[1][1], minMax[2][1])

        x1Point = [minValue,maxValue]
        x2Point = list()
        weight = self.neuronObj.getWeights()
        x2Point.append(( (-1) * weight[0] - x1Point[0] * weight[1]) / weight[2])
        x2Point.append(( (-1) * weight[0] - x1Point[1] * weight[1]) / weight[2])
        decBoundPlot.plot(x1Point, x2Point)

        decBoundPlot.set_title("Decision boundary")
        decBoundPlot.set_xlabel("Input Attribute 1 samples")
        decBoundPlot.set_ylabel("Input Attribute 2 samples")

        myplot.show()

if __name__ == '__main__':
    """
    A program main function
    """

    if len(sys.argv) != 2:
        print("ERROR: Please run the program as:")
        print("python logreg.py <filename.csv>")
        sys.exit(1)

    fileName = sys.argv[1]
    fileObject = FileOperation(fileName)

    # Read the csv data from the file
    fileObject.readFile()

    # Trained or optimizes a network
    neuronObject = NeuronUnit()
    lrObject = LogisticRegression( neuronObject, fileObject )
    combinedWeights = lrObject.batchGradientDescent()

    # Write the weights data into a weights.csv file
    fileObject.writeFile(combinedWeights)

    # Calculate a classifiers correct/Incorrect samples
    lrObject.correctIncorrectSamples()

    # Plot a graph
    figure = myplot.figure()
    lrObject.plotGraph(figure)
    lrObject.decisionBoundaryPlot(figure)
    myplot.show()
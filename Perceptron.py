import numpy
import  random
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas

"""
Global variables
"""
learningRate = 0.001
perceptron = []
testAccuracy = []
trainingAccuracy = []

#####################Perceptron Class#####################
class Perceptron:
	
	def __init__(self):
		self.weights = np.random.uniform(-0.5,0.5,785)	#Assigning ran] weights
		self.inputVector = np.zeros(785)
		self.inputVector[0] = 1.0	#Initializing input vector with bias
	
	def setWeights(self, vector):
		self.weights = np.array(vector)
			
	def setInputVector(self, vector):
		self.expectedOutput = vector[0]
		self.inputVector[1:] = vector[1:]

	def updateWeights(self,targetOutputMinusexceptedOutput):
		if targetOutputMinusexceptedOutput == 0:	#If (t-y) is zero no need to update weights just return
			return		
		self.weights += learningRate*targetOutputMinusexceptedOutput*self.inputVector
	
	def getWeights(self):
		return self.weights
				
	def getDotProduct(self):
		return np.dot(self.weights, self.inputVector)
		
	def getOutput(self):
		if np.dot(self.weights, self.inputVector) > 0:
			return 1
		else:
			return 0
#####################Perceptron Class ends#####################		

#Prints matrix 
def printConfusionMatrix(matrix):
	for i in range(0,10):
		for j in range(0,10):
			print matrix[i][j],
		print "\n"

"""		
Arguments:
inputVector: Input vector of shape (1, 785)
We get weights of all perceptrons and make an array of size (785, 10)
Compute the dot product so that we will get an array of shape (1,10)
Then index of maximum dot product is returned
"""
def getIndexofMaxDotProduct(inputVector):
	tempWeights = []
	for i in range(0, 10):
		new = []
		new = perceptron[i].getWeights()
		tempWeights.append(new)
	tempWeights = np.array(tempWeights)
	a = tempWeights.transpose()
	b = np.dot(inputVector, a)
	return np.argmax(b) 
	
"""
Arguments:
newInputData - passing inpudata as a parameter 
newTestData - data from the testing dataset
epochCycle - No of epoch cycle

We iterate through the whole training and test data set.
Calculate maximum dot product for each iteration.
Enter that entry in confusion matrix
Calculate accuracy using confusion matrix.
Display confusion matrix of test set if it is the last epoch
"""
def calculateAccuracy(newInputData, newTestData, epochCycle):
	a = np.zeros(785)
	a[0] = 1.0
	trainingConfusionMatrix = np.zeros((10,10))
	testConfusionMatrix = np.zeros((10,10))
	
	for i in range(0,60000):
		a[1:] = newInputData[i,1:]
		index = getIndexofMaxDotProduct(a)
		trainingConfusionMatrix[newInputData[i][0]][index] += 1
	for i in range(0,10000):
		a[1:] = newTestData[i,1:]
		index = getIndexofMaxDotProduct(a)
		testConfusionMatrix[newTestData[i][0]][index] += 1		
	
	testDiagonalSum = (np.sum(testConfusionMatrix.diagonal()))
	testTotalSum = (np.sum(testConfusionMatrix))
	trainingDiagonalSum = (np.sum(trainingConfusionMatrix.diagonal()))
	trainTotalSum = (np.sum(trainingConfusionMatrix))
	
	testAccuracy.append(testDiagonalSum*100/testTotalSum)
	trainingAccuracy.append(trainingDiagonalSum*100/trainTotalSum)

	if epochCycle == 49:
		printConfusionMatrix(testConfusionMatrix)
	
	return (testDiagonalSum + trainingDiagonalSum)*100/(testTotalSum + trainTotalSum)
	
def main():
	print "Reading CSV"
	"""
	Pandas read csv
	"""
	#Reading training data
	inputData = np.array(pandas.read_csv("mnist_train.csv", header = -1),np.float)
	inputData[:,1:] /= 255.0	
	print inputData.shape
	
	#Reading test data
	testData = np.array(pandas.read_csv("mnist_test.csv", header = -1),np.float)
	testData[:, 1:] /= 255.0
		
	print "Data cleaned"
	
	for index in range(0,10):
		perceptron.append(Perceptron())
	
	print "Initial accuracy: ",calculateAccuracy(inputData, testData,0)," %"
	print "Learning rate: ", learningRate
		
	#Executing for 50 epochs...
	for epoch in range(0,50):
		np.random.shuffle(inputData)	#Shuffling the input data
		for trainingSet in range(0,60000):	#For 60000 test sets
			exceptedOutput = inputData[trainingSet][0]
			
			for pIndex in range(0, 10):		#for 10 perceptrons
				perceptron[pIndex].setInputVector(inputData[trainingSet,:])
				temp = perceptron[pIndex].getOutput()
							
				target = -1	#Invalid case
				if pIndex == exceptedOutput:
					target = 1
				else:
					target = 0
				
				perceptron[pIndex].updateWeights(target-temp)	#passing (t-y) as a parameter to function

		print "Accuracy after epoch: ",epoch," ",calculateAccuracy(inputData,testData,epoch), "%"
		
	message = "Learning rate = "
	message1 = learningRate
	
	#Plotting grpah
	plt.plot(testAccuracy)
	plt.plot(trainingAccuracy)
	plt.ylabel('Accuracy')
	plt.xlabel('Epochs')
	plt.title(message+str(message1))
	plt.savefig("ResultExtra.png")
	
main()	

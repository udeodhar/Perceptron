import numpy
import  random
import time
import numpy as np
from joblib import Parallel, delayed
import multiprocessing

"""
Global variables
"""
exceptedOutput = 10
num_cores = multiprocessing.cpu_count()
confusionMatrix = []
learningRate = 0.001
inputData = []
weights = []
perceptron = []
"""
Reading CSV file
"""
import csv
with open('mnist_train.csv', 'r') as mnist_trainCSV:
	data = csv.reader(mnist_trainCSV)
	index = 0
	print "Reading data from CSV file..."
	for row in data:
		inputData.append(row)
	
#####################Perceptron Class#####################
class Perceptron:
	
	def __init__(self):
		self.weights = []
		self.inputVector = [1]
		self.expectedOutput = 10

	def setWeights(self, vector):
		self.weights = vector	
			
	def setInputVector(self, vector):
		self.expectedOutput = vector[0]
		self.inputVector = [1]
		for i in range(1,785):
			self.inputVector.append(vector[i])
	
	def updateWeights(self,targetOutputMinusexceptedOutput):
		for index in range(0,785):
			self.weights[index] += learningRate*targetOutputMinusexceptedOutput*self.inputVector[index]
	
	def getWeights(self):
		return self.weights
		
	def getDotProduct(self):
		return np.dot(np.array(self.weights), np.array(self.inputVector))
		
	def getOutput(self):
		#print "##########"
		#print "Size of weights ",len(self.weights)
		#print "Size of input vector ",len(self.inputVector)
		
		if np.dot(np.array(self.weights), np.array(self.inputVector)) > 0:
			return 1
		else:
			return 0
#####################Perceptron Class ends#####################		
		
		
"""
def initiateRandomWeights():
def calculateAccuracy():
"""
inputData2 = []
def cleanData():
	print "Cleaning data..."
	for row in range(0,60000):
		for col in range(0,785):
			if col != 0:
				inputData[row][col] = float(inputData[row][col])/255.0
			else:
				inputData[row][col] = int(inputData[row][col])
	
#def calculateAccuracy():
	
def printConfusionMatrix(matrix):
	for i in range(0,10):
		for j in range(0,10):
			print matrix[i][j],
		print "\n"
		
def initConfusionMatrix():
	###Initialize confusionMatrix (All zeros)
	for i in range(0,10):
		new = []	
		for j in range(0,10):
			new.append(0)
		confusionMatrix.append(new)
	
def calculateAccuracy():
	rightCount = 0;
	for i in range (0,60000):
		exceptedOutput = inputData[i][0]
		maxDotProduct = 0
		indexOfMaxDotProduct = 10
		for j in range(0,10):
			perceptron[j].setInputVector(inputData[i])
			temp = perceptron[j].getDotProduct()
			if maxDotProduct < temp:
				maxDotProduct = temp
				indexOfMaxDotProduct = j
		if exceptedOutput == indexOfMaxDotProduct:
			rightCount += 1
	return (rightCount/60000.0)*100
	"""
	noOfDiagonalOnes = 0
	totalNumberOfOnes = 0;
	for i in range(0,10):
		for j in range(0,10):
			if confusionMatrix[i][j] == 1:
				totalNumberOfOnes += 1
				if i == j:
					noOfDiagonalOnes += 1
					
	if totalNumberOfOnes == 0:
		return 0
	return float(noOfDiagonalOnes)/float(totalNumberOfOnes)	
	"""

def processInput(tuple):
	(pIndex, exceptedOutput) = tuple
	perceptron[pIndex].setInputVector(inputData[trainingSet])
	temp = perceptron[pIndex].getOutput()
		
	#update confusion matrix
	#if temp == 1:
	#	confusionMatrix[pIndex][exceptedOutput] = 1
		
	if ((temp == 1) and (pIndex != exceptedOutput)) or (temp == 0) and (pIndex == exceptedOutput):
		if temp == 1:
			perceptron[pIndex].updateWeights(-1)
		else:
			perceptron[pIndex].updateWeights(1)

	
def main():
	cleanData()
	print "Data cleaned"
	#time.sleep(3)
	
	"""
	Calculate accuracy
	Detect the "wrong" perceptrons
	update weights of those perceptrons
	"""
		
	
	for index in range(0,10):
		perceptron.append(Perceptron())
		weight = []
		for i in range(0,785):
			weight.append(random.uniform(-0.5,0.5))
		perceptron[index].setWeights(weight)
	
	"""
	print inputData[0]
	perceptron[0].setInputVector(inputData[0])
	print perceptron[0].getOutput()
	"""
	
	initConfusionMatrix()
	#print confusionMatrix
	printConfusionMatrix(confusionMatrix)
	#calculateAccuracy for epoch 0
	
	print "Initial accuracy: ",calculateAccuracy()," %"


	
	for epoch in range(0,50):
		np.random.shuffle(inputData)
		initConfusionMatrix()
		for trainingSet in range(0,1000):
			exceptedOutput = inputData[trainingSet][0]
			#inputs = range(0,10)
			#results = Parallel(n_jobs=num_cores)(delayed(processInput)(i,exceptedOutput) for i in inputs)		
			#tuple = (i, exceptedOutput)
			#results = Parallel(n_jobs=10, backend = "threading")(map(delayed(processInput),tuple))
			
			#print "Training set ", trainingSet
			
			for pIndex in range(0, 10):
				perceptron[pIndex].setInputVector(inputData[trainingSet])
				temp = perceptron[pIndex].getOutput()
			
				#update confusion matrix
				#if temp == 1:
			#	confusionMatrix[pIndex][exceptedOutput] = 1
			
				if ((temp == 1) and (pIndex != exceptedOutput)) or (temp == 0) and (pIndex == exceptedOutput):
					if temp == 1:
						perceptron[pIndex].updateWeights(-1)
					else:
						perceptron[pIndex].updateWeights(1)
			
		print "Accuracy after epoch: ",epoch," ",calculateAccuracy(), "%"
		#print "Epoch ",epoch
#		time.sleep(5)
	print "Accuracy after epoch",epoch," ",calculateAccuracy()," %"

main()
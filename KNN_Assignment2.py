import csv
import random
import math

# Load dataset from file
def loadDataset(filename, split):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        dataset.pop(0)  # remove header row
        for i in range(len(dataset)):
            for j in range(4):
                dataset[i][j] = float(dataset[i][j])
        random.shuffle(dataset)
        split_index = int(len(dataset) * split)
        trainingSet = dataset[:split_index]
        testSet = dataset[split_index:]
        return trainingSet, testSet

#4- Calculate the Euclidean distance between two instances
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for i in range(length):
        distance += pow((instance1[i] - instance2[i]), 2)
    return math.sqrt(distance)

# Get k nearest neighbors for a test instance
def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for i in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[i], length)
        distances.append((trainingSet[i], dist))
    distances.sort(key=lambda x: x[1]) # Sort distances to get nearest k distances
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors

#3- Predict the class for a test instance (If there is a tie in the class predicted by the k-nn, then among the classes that have the same number of votes, you should pick the one that comes first in the Train file.)
def getClass(neighbors):
    classVotes = {}  # 
    for i in range(len(neighbors)):
        response = neighbors[i][-1]  # assign the class label of the kth neighbor to the variable response
        if response in classVotes:
            classVotes[response] += 1 # in case response already exits increment count by 1
        else:
            classVotes[response] = 1 # in case response not exit new key-value pair is created in the dictionary with the class as the key and its count set to 1.
    sortedVotes = sorted(classVotes.items(), key=lambda x: x[1], reverse=True) # sort classvote in descending order based on their count values, before sorting the data, the first one in the train is put first
    return sortedVotes[0][0] # return highest class

# 2- Normalize the dataset
def normalizeDataset(dataset):
    for i in range(len(dataset[0])-1):
        column = [row[i] for row in dataset] # Each feature column should be normalized separately from all other features.
        mean = sum(column) / float(len(column))
        variance = sum([pow(x-mean,2) for x in column]) / float(len(column)-1)  # (variance)^2 = sigma(value-mean)^2 / total_num
        std = math.sqrt(variance)
        for row in dataset:
            row[i] = (row[i] - mean) / std  # f(v) = (v - mean) / std

# Run KNN algorithm for different values of k and print results
def runKNN(dataset, k_values):
    print("KNN Classifier Results:")
    print("-----------------------")
    for k in k_values:
        correct = 0 # The number of correctly classified test instances.
        for i in range(len(dataset)):
            neighbors = getNeighbors(dataset, dataset[i], k)
            predictedClass = getClass(neighbors)
            if predictedClass == dataset[i][-1]:
                correct += 1
        accuracy = (correct / float(len(dataset))) * 100.0
        print("k = {}: Correct = {}, Total = {}, Accuracy = {:.2f}%".format(k, correct, len(dataset), accuracy)) # Dataset is The total number of instances in the test set.

# Load and prepare data
filename = 'D:/Study Materials/Year 4/Term 2/Machine Learning/Assignments/Assignment 2/BankNote_Authentication.csv'
split = 0.7  #1- Divide your data into 70% for training and 30% for testing.
trainingSet, testSet = loadDataset(filename, split)
normalizeDataset(trainingSet)
normalizeDataset(testSet)

# Define values of k to test
k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Run KNN algorithm and print results
runKNN(testSet, k_values)

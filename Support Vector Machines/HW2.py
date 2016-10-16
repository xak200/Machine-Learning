import numpy as np
import matplotlib.pyplot as plt
import math

#problem 1
def plotSVMtrainData():
    plt.scatter(2, 2, c = 'w')
    plt.scatter(4, 4, c = 'w')
    plt.scatter(4, 0, c = 'w')
    plt.scatter(0, 0, c = 'b')
    plt.scatter(2, 0, c = 'b')
    plt.scatter(0, 2, c = 'b')
    x1 = [-1, 3]
    y1 = [3, -1]
    x2 = [5, -1]
    y2 = [-1, 5]
    x3 = [4, -1]
    y3 = [-1, 4]
    x4 = [1, 2]
    y4 = [1, 2]
    negative, = plt.plot(x1, y1, 'r--', label='negative support vector')
    positive, = plt.plot(x2, y2, 'g--', label='positive support vector')
    decision, = plt.plot(x3, y3, label='decision hyperplane')
    weight, = plt.plot(x4, y4, label='weight vector')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim(-1, 8)
    plt.ylim(-1, 8)
    plt.legend(handles=[negative, positive, decision, weight])
    plt.show()

#plotSVMtrainData()

#problem 3
def mySet(seq):
    # order preserving
    checked = []
    for e in seq:
        if e not in checked:
            checked.append(e)
    return checked

def splitFile():
    trainingSplit = 4000
    emailCount = 0
    f = open('ps1_data/spam_train.txt', 'r')
    t = open('ps1_data/new_train.txt', 'w')
    v = open('ps1_data/validate.txt', 'w')
    for email in f:
        emailCount += 1
        if emailCount <= trainingSplit:
            t.write(email)
        elif emailCount > trainingSplit:
            v.write(email)
    f.close()
    t.close()
    v.close()
    return

def createTotalDict():
    emailCount = 0
    totalDict = {}
    t = open('ps1_data/new_train.txt', 'r')
    for email in t:
        emailCount += 1
        email = mySet(email.split())  ##unique words
        for word in email:
            if word != str(1) and word != str(0):
                if word in totalDict:
                    value = totalDict[word]
                    value += 1
                    totalDict[word] = value
                else:
                    totalDict[word] = 1
    t.close()
    return totalDict

def createSpamDict(totalDict):
    spamThreshold = 30
    spamDict = {}
    for x in totalDict:
        if totalDict[x] >= spamThreshold:
            spamDict[x] = totalDict[x]
    return spamDict

#read from file, convert each email into featurevector data based on spamdict and result vector
def featureVectors(spamDict, fileName, size):
    counter = 0
    featureVectorList = np.zeros((size, len(spamDict)))
    resultVector = np.zeros((size))
    t = open('ps1_data/' + fileName, 'r')
    for email in t:
        i = 0
        if email[0] == str(0):
            resultVector[counter] = -1
        else:
            resultVector[counter] = 1
        email = list(mySet(email.split()))
        spamWords = np.zeros((len(spamDict)))
        for word in spamDict:
            if word in email:
                spamWords[i] = 1
            else:
                spamWords[i] = 0
            i += 1
        featureVectorList[counter] = spamWords
        counter += 1
    t.close()
    return (featureVectorList, resultVector)

def spamDictLength(totalDict):
    spamThreshold = 30
    counter = 0
    for x in totalDict:
        if totalDict[x] >= spamThreshold:
            counter += 1
    return counter

def pegasos_svm_train(data, result, spamDict, myLambda):
    t = 0
    numMistakes = 0
    totalHingeLoss = 0
    maxPasses = 20
    svmObjective = []
    w = np.zeros(len(spamDict))
    for i in range(maxPasses):
        for j in range(len(data)):
            t += 1
            eta = 1/(t*myLambda)
            prediction = result[j] * np.dot(w, data[j])
            #use support vectors
            if (prediction) < 1:
                w = ((1-(eta*myLambda)) * w) + (eta * result[j] * data[j])
            else:
                w = ((1-(eta*myLambda)) * w)
            #use classifier
            if (prediction) < 0:
                numMistakes +=1
        hingeLoss = 0
        for k in range(len(data)):
            hingeLoss += max(0, 1 - result[i]*np.dot(data[i], w))
        hingeLoss = hingeLoss / len(result)
        totalHingeLoss += hingeLoss
        svmObjective.append((t, ((myLambda/2) * math.pow(np.linalg.norm(w), 2)) + hingeLoss))
    return (w, svmObjective, numMistakes/(maxPasses*len(data)), totalHingeLoss/maxPasses)

def pegasos_svm_test(w, data, result):
    errorVector = np.zeros(len(result))
    multiple = np.multiply(result, np.dot(data, w))
    for i in range(len(result)):
        if multiple[i] < 0:
            errorVector[i] = 1
        else:
            errorVector[i] = 0
    testError = errorVector.sum() * 1. / len(result)
    return testError

def iterateRegConstant(myLambdaExp, spamDict, trainData, trainResult, validationData, validationResult):
    plotData = []
    for i in myLambdaExp:
        myLambda = math.pow(2, i)
        w, svmObjective, trainError, hingeLoss = pegasos_svm_train(trainData, trainResult, spamDict, myLambda)
        validationError = pegasos_svm_test(w, validationData, validationResult)
        print('log base 2 lambda', i, 'validation err', validationError)
        plotData.append((i, trainError, hingeLoss, validationError))
    plot4Vector(plotData)

def runIt():
    splitFile()
    totalDict = createTotalDict()
    print('total vocablist', len(totalDict))
    print('number of words in spamDict', spamDictLength(totalDict))
    spamDict = createSpamDict(totalDict)
    #3a
    trainData, trainResult = featureVectors(spamDict, 'new_train.txt', 4000)
    w, svmObjective, trainError, hingeLoss = pegasos_svm_train(trainData, trainResult, spamDict, math.pow(2, -5))
    plot2Vector(svmObjective)
    #3b
    validationData, validationResult = featureVectors(spamDict, 'validate.txt', 1000)
    #3c
    myLambdaExp = []
    myLambdaExp.extend(range(-9, 2))
    iterateRegConstant(myLambdaExp, spamDict, trainData, trainResult, validationData, validationResult)
    testData, testResult = featureVectors(spamDict, 'spam_test.txt', 5000)
    w, svmObjective, trainError, hingeLoss = pegasos_svm_train(trainData, trainResult, spamDict, math.pow(2, -8))
    print('testerror = ', pegasos_svm_test(w, testData, testResult))
    print('numsupportvectors=', findNumSupportVectors(w, trainData, trainResult))

def findNumSupportVectors(w, data, result):
    numSupportVectors = 0
    for j in range(len(data)):
        if (result[j] * np.dot(w, data[j])) <= 1:
            numSupportVectors +=1
    return numSupportVectors

def plot2Vector(vector):
    x, y = zip(*vector)
    plt.plot(x, y)
    plt.ylabel('f(wt)')
    plt.xlabel('t')
    plt.show()

def plot4Vector(vector):
    a, b, c, d = zip(*vector)
    plt.ylabel('Errors and Losses')
    plt.xlabel('Log(base 2) lambda')
    plt.ylim(0, 0.8)
    avgTrainingError, = plt.plot(a, b, 'r--', label='average training error')
    avgHingeLoss, = plt.plot(a, c, 'b--', label='average hinge loss')
    avgValidationError, = plt.plot(a, d, 'g--', label='average validation error')
    plt.legend(handles=[avgTrainingError, avgHingeLoss, avgValidationError])
    plt.show()

runIt()
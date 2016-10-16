import numpy as np
import matplotlib.pyplot as plt

def mySet(seq):
    # order preserving
    checked = []
    for e in seq:
        if e not in checked:
            checked.append(e)
    return checked

def splitFile():
    emailCount = 0
    f = open('ps1_data/spam_train.txt', 'r')
    t = open('ps1_data/new_train.txt', 'w')
    v = open('ps1_data/validate.txt', 'w')
    for email in f:
        emailCount += 1
        if emailCount <= 4000:
            t.write(email)
        elif emailCount > 4000:
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
                ##        if emailCount % 100 == 0:
                ##            print(emailCount)
    t.close()
    return totalDict

def createSpamDict(totalDict):
    spamDict = {}
    for x in totalDict:
        if totalDict[x] >= 30:
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
        resultVector[counter] = email[0]
        email = list(mySet(email.split()))
        spamWords = np.zeros((len(spamDict)))
        #featureVectorList[counter] = email[0]
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
    counter = 0
    for x in totalDict:
        if totalDict[x] >= 30:
            counter += 1
            ##            print(x + ' ' + str(totalDict[x]))
    return counter

def perceptron_train(data, result, N, maxPasses):
    convert = lambda x: 1 if x >= 0 else 0
    w = np.zeros(len(spamDict))
    kNumMistakes = 0
    numPasses = 0
    numErrorsInLastCycle = 1
    while numErrorsInLastCycle != 0:
        numErrorsInLastCycle = 0
        for i in range(0, N - 1):
            featureVector = data[i]
            predicted = np.dot(w, featureVector)
            error = result[i] - convert(predicted)
            if error != 0:
                kNumMistakes += 1
                numErrorsInLastCycle += 1
                w += error * featureVector
        numPasses += 1
        if numPasses == maxPasses:
            break
        #print(numPasses, kNumMistakes, numErrorsInLastCycle)
    return (w, kNumMistakes, numPasses)

def perceptron_test(w, data, result):
    convert = lambda x: 1 if x >= 0 else 0
    kNumMistakes = 0
    for i in range(0, len(data) - 1):
        featureVector = data[i]
        predicted = np.dot(w, featureVector)
        error = result[i] - convert(predicted)
        if error != 0:
            kNumMistakes += 1
    testError = kNumMistakes/len(data)
    return testError

def findMinMax(w, spamDict):
    dtype = [('weight', float), ('word', str)]
    spamList = []
    for key in spamDict:
        spamList.append(key)
    weightedSpamList = np.column_stack((w, spamList))
    sortedSpamList = sorted(weightedSpamList, key=lambda x: float(x[0]))
    mostSpam = sortedSpamList[-15:]
    leastSpam = sortedSpamList[:15]
    return(mostSpam, leastSpam)

def averagePerceptronTrain(data, result, N, maxPasses):
    convert = lambda x: 1 if x >= 0 else 0
    w = np.zeros(len(spamDict))
    totalW = np.zeros(len(spamDict))
    kNumMistakes = 0
    numPasses = 0
    numErrorsInLastCycle = 1
    while numErrorsInLastCycle != 0:
        numErrorsInLastCycle = 0
        for i in range(0, N - 1):
            featureVector = data[i]
            predicted = np.dot(w, featureVector)
            error = result[i] - convert(predicted)
            if error != 0:
                kNumMistakes += 1
                numErrorsInLastCycle += 1
                w += error * featureVector
            totalW += w
        numPasses += 1
        if numPasses == maxPasses:
            break
    averageW = totalW / (numPasses * len(data))
    return (averageW, kNumMistakes, numPasses)

def iterateTests(N):
    normalError = []
    avgError = []
    normalPasses = []
    for i in N:
        w, k, myIter = perceptron_train(trainData, trainResult, i, 6)
        normalPasses.append(myIter)
        print('error on traindata for', i, 'is', perceptron_test(w, trainData, trainResult))
        normalError.append(perceptron_test(w, validationData, validationResult))
        w, k, myIter = averagePerceptronTrain(trainData, trainResult, i, 6)
        print('error on avg perc traindata for', i, 'is', perceptron_test(w, trainData, trainResult))
        avgError.append(perceptron_test(w, validationData, validationResult))
    return (normalError, avgError, normalPasses)

splitFile()
totalDict = createTotalDict()
print('total vocablist', len(totalDict))
print('number of words in spamDict', spamDictLength(totalDict))
spamDict = createSpamDict(totalDict)
trainData, trainResult = featureVectors(spamDict, 'new_train.txt', 4000)
validationData, validationResult = featureVectors(spamDict, 'validate.txt', 1000)

N = [100, 200, 400, 800, 2000, 4000]
e1, e2, p1 = iterateTests(N)

plt.subplot(211)
plt.plot(N, e1, 'r--', N, e2, 'b--')
plt.ylabel('Error rates')
plt.xlabel('Training set size')

plt.subplot(212)
plt.plot(N, p1, 'r--')
plt.ylabel('Number of passes')
plt.xlabel('Training set size')
plt.ylim(0, 12)
plt.show()

trainData, trainResult = featureVectors(spamDict, 'spam_train.txt', 5000)
validationData, validationResult = featureVectors(spamDict, 'spam_test.txt', 5000)
w, k, myIter = averagePerceptronTrain(trainData, trainResult, 5000, 6)
print(perceptron_test(w, validationData, validationResult))
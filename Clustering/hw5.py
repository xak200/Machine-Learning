def randIndex(truth, predicted):
    """
    The function is to measure similarity between two label assignments
    truth: ground truth labels for the dataset (1 x 1496)
    predicted: predicted labels (1 x 1496)
    """
    if len(truth) != len(predicted):
        print("different sizes of the label assignments")
        return -1
    elif (len(truth) == 1):
        return 1
    sizeLabel = len(truth)
    agree_same = 0
    disagree_same = 0
    count = 0
    for i in range(sizeLabel - 1):
        for j in range(i + 1, sizeLabel):
            if ((truth[i] == truth[j]) and (predicted[i] == predicted[j])):
                agree_same += 1
            elif ((truth[i] != truth[j]) and (predicted[i] != predicted[j])):
                disagree_same += 1
            count += 1
    return (agree_same + disagree_same) / float(count)


# Code Sample
import scipy.cluster.hierarchy as sch
import numpy as np
import pylab as pl
import random
from scipy import stats

# Plot dendogram and cut the tree to find resulting clusters
fig = pl.figure()
data = np.array([[1, 2, 3], [1, 1, 1], [5, 5, 5]])
datalable = ['first', 'second', 'third']
hClsMat = sch.linkage(data, method='complete')  # Complete clustering
sch.dendrogram(hClsMat, labels=datalable, leaf_rotation=45)
fig.show()
fig.savefig('toydata.png')
resultingClusters = sch.fcluster(hClsMat, t=3, criterion='distance')


# Your code starts from here ....

# 1.
# Scaling min max
# STUDENT CODE TODO

# read file into 2d-array
def readFiles():
    c = open('data/dataCereal-grains-pasta.txt', 'r')
    fo = open('data/dataFats-oils.txt', 'r')
    fs = open('data/dataFinfish-shellfish.txt', 'r')
    v = open('data/dataVegetables.txt', 'r')

    dataArray = []
    labelArray = []
    files = {c:'cereal', fo:'fats', fs:'fish', v:'vegetables'}

    for file, foodType in files.items():
        for line in file:
            newLine = line.split('^')
            newLine.pop()  # get rid of new line character
            dataArray.append(newLine)
            labelArray.append(foodType)
        file.close()
    return (dataArray, labelArray)


def minsAndMaxes(dataArray):
    mins = np.zeros(151)
    million = 1000000.0
    maxes = np.zeros(151)
    #initialize min and max arrays
    counter = 0
    for ele in dataArray[0]:
        if (counter == 0):
            counter += 1
            continue
        mins[counter] = million
        maxes[counter] = float(maxes[counter])
        counter += 1

    #calculate mins and maxes
    for featureVector in dataArray:
        counter = 0
        for feature in featureVector:
            if (counter == 0):
                counter += 1
                continue
            if (float(feature) > maxes[counter]):
                maxes[counter] = float(feature)
            if (float(feature) < mins[counter]):
                mins[counter] = float(feature)
            counter += 1

    return (mins, maxes)

# replace each feature value with normalized value
def normalize(mins, maxes, data, labels):
    normalizedData = []
    cereals = []
    cerealItems = []
    fats = []
    fish = []
    vegetables = []
    itemCounter = 0
    for item in data:
        featureCounter = 0
        normalizedFeatures = []
        newDesc = []
        for feature in item:
            if (featureCounter == 0):
                newDesc.append(labels[itemCounter])
                newDesc.append(feature)
            else:
                if (maxes[featureCounter] == mins[featureCounter]):
                    normalized = 0.0
                else:
                    normalized = (float(feature) - mins[featureCounter]) / float(maxes[featureCounter] - mins[featureCounter])
                normalizedFeatures.append(normalized)
            featureCounter += 1
        normalizedData.append(normalizedFeatures)
        if (labels[itemCounter] == 'cereal'):
            cereals.append(normalizedFeatures)
            cerealItems.append(newDesc)
        if (labels[itemCounter] == 'fats'):
            fats.append(normalizedFeatures)
        if (labels[itemCounter] == 'fish'):
            fish.append(normalizedFeatures)
        if (labels[itemCounter] == 'vegetables'):
            vegetables.append(normalizedFeatures)
        itemCounter += 1
    return (normalizedData, cerealItems, cereals, fats, fish, vegetables)


def run1():
    data, labels = readFiles()
    mins, maxes = minsAndMaxes(data)
    normalized, cerealItems, cereals, fats, fish, veg = normalize(mins, maxes, data, labels)
    return (normalized, cerealItems, labels, cereals, fats, fish, veg)

normalized, cerealItems, groundTruth, cereals, fats, fish, veg = run1()

# 2.
# K-means http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
# STUDENT CODE TODO
from sklearn.cluster import KMeans

def run2(normalized):
    X = normalized
    estimator = KMeans(n_clusters=4)
    estimator.fit(X)
    return (estimator)

estimator = run2(normalized)

# 3.
# Compute Rand Index
# STUDENT CODE TODO
def run3():
    prediction = estimator.labels_
    print ('3. randIndex = ', randIndex(groundTruth, prediction))

run3()


# 4.
# Examining K-mean objective
# STUDENT CODE TODO
def run4():
    X = normalized
    estimator = KMeans(n_clusters=4, n_init=1)
    for i in range(20):
        estimator.fit(X)

run4()

# 5.
# Dendogram plot
# Dendogram - http://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.cluster.hierarchy.dendrogram.html
# Linkage - http://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.cluster.hierarchy.linkage.html
# STUDENT CODE TODO
def run5():
    fig = pl.figure()

    randomCereals = random.sample(cereals, 30)
    randomFats = random.sample(fats, 30)
    randomFish = random.sample(fish, 30)
    randomVeg = random.sample(veg, 30)

    data = np.concatenate((randomCereals, randomFats, randomFish, randomVeg))
    datalabel = (['l'] * 30) + (['t'] * 30) + (['i'] * 30) + (['j'] * 30)       #cereal, fats, fish, veg

    hClsMat = sch.linkage(data, method='complete')
    sch.dendrogram(hClsMat, labels=datalabel, leaf_font_size=6)
    fig.savefig('plot.png')
    fig.show()

run5()

# 6.
# Hierarchical clustering
# SciPy's Cluster - http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html#scipy.cluster.hierarchy.fcluster
# STUDENT CODE TODO
def run6():
    hClsMat = sch.linkage(normalized, method='complete')
    resultingClusters = sch.fcluster(hClsMat,t=3.8, criterion='distance')
    print('6. randIndex, t = 3.8:', randIndex(groundTruth, resultingClusters))

    maxRI = 0
    maxThreshold = 0
    for i in range(0, 38):
        resultingClusters = sch.fcluster(hClsMat,t=(i/10), criterion='distance')
        temp = randIndex(groundTruth, resultingClusters)
        if (temp > maxRI):
            maxRI = temp
            maxThreshold = i
    print('6. Best threshold, RI =',maxThreshold/10, maxRI)
    print('6. Unique clusters:', set(resultingClusters))

run6()

# 7.
# K-means for Sub-cluster
# STUDENT CODE TODO

def run7(cereals, cerealItems):
    numClusters = [5, 10, 25, 50, 75]
    for i in numClusters:
        estimator = KMeans(n_clusters=i, n_init=1)
        estimator.fit(cereals)
        prediction = estimator.labels_
        mode, count = stats.mode(prediction, axis=None)
        modeNum = mode[0]
        countNum = count[0]
        modeIndices = []
        count = 0
        for j in prediction:
            if (j == modeNum):
                modeIndices.append(count)
            count += 1
        if (len(modeIndices) < 10):
            for k in modeIndices:
                print('Number of clusters, Cluster size, Item name: ', i, ',', countNum, ',', cerealItems[k][1])
        else:
            randomIndices = random.sample(modeIndices, 10)
            for index in randomIndices:
                print('Number of clusters, Cluster size, Item name: ', i, ',', countNum, ',', cerealItems[index][1])

run7(cereals, cerealItems)
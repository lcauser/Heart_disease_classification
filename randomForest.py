import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Run an experiment with a decision tree.
def experiment(data, numTrees, nodes):
    # Shuffle the data
    shuffledData = data.sample(frac=1)
    
    # Split data
    Y = shuffledData['target']
    X = shuffledData.drop(['target'], axis=1)
    XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=0.2)
    
    # Make the model and fit it
    model = RandomForestClassifier(numTrees, max_leaf_nodes=nodes,
                                   random_state=1)
    model.fit(XTrain, YTrain)
    
    # Test the model
    YPred = model.predict(XTest)
    return accuracy_score(YTest, YPred)

numExperiments = 1000

# Load the file
fileName = "heart.csv"
data = pd.read_csv(fileName)
data = data.drop(['chol', 'fbs'], axis=1)

# Split data into testing and training
Y = data['target']
X = data.drop(['target'], axis=1)
XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=0.2)

#%% Train a random forest on 5 leaf nodes
numTrees = np.linspace(5, 200, 40).astype(np.int)
numLeafNodes = 5
accMeans = []
accStds = []
for numTree in numTrees:
    accs = []
    for ex in range(numExperiments):
        # Run an experiment
        accs.append(experiment(data, numTree, numLeafNodes))
    accMeans.append(np.mean(accs))
    accStds.append(np.std(accs))
    
#%% Plot
plt.figure()
plt.errorbar(numTrees, accMeans, yerr=accStds, marker='o', linestyle='--')
plt.xlabel("# Trees")
plt.ylabel("Accuracy")

#%% Train a random forest on 100 trees
numTrees = 100
numLeafNodes = 10
accMeans = []
accStds = []
for numLeafNode in range(2, numLeafNodes+1):
    accs = []
    for ex in range(numExperiments):
        # Run an experiment
        accs.append(experiment(data, numTrees, numLeafNode))
    accMeans.append(np.mean(accs))
    accStds.append(np.std(accs))

plt.figure()
plt.errorbar(range(2, numLeafNodes+1), accMeans, yerr=accStds, marker='o', linestyle='--')
plt.xlabel("Leaf Nodes")
plt.ylabel("Accuracy")

#%% Train a random forest on 100 trees, 8 nodes
numTrees = 100
numLeafNodes = 8
numExperiments = 10000

accs = []
for ex in range(numExperiments):
    # Run an experiment
    accs.append(experiment(data, numTrees, numLeafNodes))

accuracy = np.mean(accs)
error = np.std(accs)

#%% Bin Data and plot histogram
bins = np.linspace(0, 1, 51)
nums = np.digitize(accs, bins)
plt.hist(accs, bins, density=True)
plt.xlabel('Acceptance')
plt.ylabel('Probability Density')
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import cross_val_score

# Run an experiment with a decision tree.
def experiment(data, nodes):
    # Shuffle the data
    shuffledData = data.sample(frac=1)
    
    # Split data
    Y = shuffledData['target']
    X = shuffledData.drop(['target'], axis=1)
    XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=0.2)
    
    # Make the model and fit it
    model = DecisionTreeClassifier(max_leaf_nodes=nodes, random_state=1)
    model.fit(XTrain, YTrain)
    
    # Test the model
    YPred = model.predict(XTest)
    return accuracy_score(YTest, YPred)

numExperiments = 1000
finalTests = 10000
maxNodes = 20

# Load the file
fileName = "heart.csv"
data = pd.read_csv(fileName)
data = data.drop(['chol', 'fbs'], axis=1)

#%% Test experiments
# Run experiments on varying nodes
accsMean = []
accsStd = []
for nodes in range(2, maxNodes+1):
    accs = []
    for ex in range(numExperiments):
        # Run an experiment
        accs.append(experiment(data, nodes))
    
    accsMean.append(np.mean(accs))
    accsStd.append(np.std(accs))

 
# Plot CV data
plt.figure()
plt.errorbar(range(2, maxNodes+1), accsMean, yerr=accsStd, marker='o', linestyle='--')
plt.xlabel("Leaf Nodes")
plt.ylabel("Accuracy")

#%% Find the optimal number of nodes and run more experiements to determine
#   accuracy
#nodes = np.argmax(accsMean) + 2
nodes = 5
accs = []
for ex in range(finalTests):
    # Run an experiment
    accs.append(experiment(data, nodes))
accuracy = np.mean(accs)
error = np.std(accs)

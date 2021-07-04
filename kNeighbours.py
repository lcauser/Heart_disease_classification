import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# Run an experiment with a decision tree.
def experiment(data, n=5):
    # Shuffle the data
    shuffledData = data.sample(frac=1)
    
    # Split data
    Y = shuffledData['target']
    X = shuffledData.drop(['target'], axis=1)
    XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=0.2)
    
    # Make the model and fit it
    model = KNeighborsClassifier(n_neighbors=n)
    model.fit(XTrain, YTrain)
    
    # Test the model
    YPred = model.predict(XTest)
    return accuracy_score(YTest, YPred)

numExperiments = 1000
finalTests = 10000
maxNeighbours = 10

# Load the file
fileName = "heart.csv"
data = pd.read_csv(fileName)
data = data.drop(['chol', 'fbs'], axis=1)

#%% Find optimal number of 
accsMean = []
accsStd = []
for n in range(1, maxNeighbours+1):
    accs = []
    for ex in range(numExperiments):
        # Run an experiment
        accs.append(experiment(data, n))
    
    accsMean.append(np.mean(accs))
    accsStd.append(np.std(accs))

 
# Plot CV data
plt.figure()
plt.errorbar(range(1, maxNeighbours+1), accsMean, yerr=accsStd, marker='o', linestyle='--')
plt.xlabel("k")
plt.ylabel("Accuracy")

#%%
k = np.argmax(accsMean)+1
accs = []
for ex in range(finalTests):
    accs.append(experiment(data, k))
accuracy = np.mean(accs)
error = np.std(accs)
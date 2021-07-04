import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

# Run an experiment with a decision tree.
def experiment(data, kernel='poly', d=2):
    # Shuffle the data
    shuffledData = data.sample(frac=1)
    
    # Split data
    Y = shuffledData['target']
    X = shuffledData.drop(['target'], axis=1)
    XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=0.2)
    
    # Make the model and fit it
    model = SVC(kernel=kernel, gamma='auto', degree=d, random_state=1)
    model.fit(XTrain, YTrain)
    
    # Test the model
    YPred = model.predict(XTest)
    return accuracy_score(YTest, YPred)

numExperiments = 1000

# Load the file
fileName = "heart.csv"
data = pd.read_csv(fileName)
data = data.drop(['chol', 'fbs'], axis=1)

accs = []
for ex in range(numExperiments):
    accs.append(experiment(data, 'linear'))
accuracy = np.mean(accs)
error = np.std(accs)
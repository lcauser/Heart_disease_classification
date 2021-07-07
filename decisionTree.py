import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

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
    return accuracy_score(YTest, YPred), model

numExperiments = 1000
finalTests = 10000
maxNodes = 20

# Load the file
fileName = "heart.csv"
data = pd.read_csv(fileName)

# Split data into caterogical
cats = ['cp', 'exang', 'slope', 'ca', 'thal', 'restecg']
ohe = OneHotEncoder(categories='auto')
feature_arr = ohe.fit_transform(data[cats]).toarray()
feature_labels = ohe.categories_
label = []
for i in range(len(cats)):
    for lbl in feature_labels[i]:
        label.append(cats[i] + str(lbl))
    
    data = data.drop(cats[i], axis=1)
data[label] = feature_arr
    

#%% Test experiments
# Run experiments on varying nodes
accsMean = []
accsStd = []
for nodes in range(2, maxNodes+1):
    accs = []
    for ex in range(numExperiments):
        # Run an experiment
        accs.append(experiment(data, nodes)[0])
    
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
nodes = 6
accs = []
for ex in range(finalTests):
    # Run an experiment
    accs.append(experiment(data, nodes)[0])
accuracy = np.mean(accs)
error = np.std(accs)

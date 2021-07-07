import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Run an experiment with a decision tree.
def experiment(data):
    # Shuffle the data
    shuffledData = data.sample(frac=1)
    
    # Split data
    Y = shuffledData['target']
    X = shuffledData.drop(['target'], axis=1)
    XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=0.2)
    
    # Make the model and fit it
    model = LogisticRegression(random_state=1, max_iter=10000)
    model.fit(XTrain, YTrain)
    
    # Test the model
    YPred = model.predict(XTest)
    return accuracy_score(YTest, YPred)

numExperiments = 10000

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
#data = data.drop(['chol', 'fbs'], axis=1)

accs = []
for ex in range(numExperiments):
    accs.append(experiment(data))
accuracy = np.mean(accs)
error = np.std(accs)
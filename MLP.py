import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load in the data
fileName = "heart.csv"
data = pd.read_csv(fileName)
data = data.drop(['chol', 'fbs'], axis=1)

#%% The first layer of the NN should be feature columns; these are the start
### of a pipeline to transform the data into appropiate formats for a NN.
feature_columns = []

# Numeric columns
numerical_columns = ['age', 'trestbps', 'thalach', 'oldpeak', 'ca']
for col in numerical_columns:
    feature_columns.append(tf.feature_column.numeric_column(col))

# Categorical data
categorical_columns = ['sex', 'cp', 'restecg', 'exang', 'slope', 'ca',
                       'thal']
for col in categorical_columns:
    vals = data[col].unique() # Find the unique values
    cat = tf.feature_column.categorical_column_with_vocabulary_list(col, vals)
    cat_one_hot = tf.feature_column.indicator_column(cat)
    feature_columns.append(cat_one_hot)

#%% Transform the dataset into a Tensorflow dataset.
def tf_dataset(df, batch_size=32):
    df = df.copy() # Let's not change the original dataset
    labels = df.pop("target") # Pull out the labels
    return tf.data.Dataset.from_tensor_slices((dict(df), labels)) \
                .shuffle(buffer_size=len(df)).batch(batch_size)

# Split data into test and train
train, test = train_test_split(data, test_size=0.2)
train = tf_dataset(train)
test = tf_dataset(test)

#%% Construct the NN; feature columns first, binary last
def model(hidden_nodes):
    return tf.keras.models.Sequential([
        tf.keras.layers.DenseFeatures(feature_columns=feature_columns),
        tf.keras.layers.Dense(units=hidden_nodes, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])

mod = model(64)
mod.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# Train it
history = mod.fit(train, validation_data=test, epochs=200)

# Measure metrics
acc = mod.evaluate(test)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'])


#%% Run Experiments with varying number of nodes
epochs = 300
experiments = 10
nodes = np.linspace(10, 100, 10).astype(int)

def experiment(data, nodes, epochs):
    # Split data randomly
    train, test = train_test_split(data, test_size=0.2)
    train = tf_dataset(train)
    test = tf_dataset(test)
    
    mod = model(nodes)
    mod.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])
    mod.fit(train, epochs=epochs, verbose=0)
    return mod.evaluate(test, verbose=0)[1]

accsMean = []
accsStd = []
for node in nodes:
    accs = []
    for ex in range(experiments):
        accs.append(experiment(data, node, epochs))
        print("Nodes = "+str(node)+ " Experiment="+str(ex+1)+"/"+str(experiments))
    accsMean.append(np.mean(accs))
    accsStd.append(np.std(accs))

plt.figure()
plt.errorbar(nodes, accsMean, yerr=accsStd, marker='o', linestyle='--')
plt.xlabel("Neurons")
plt.ylabel("Accuracy")

#%% Run experiments with just 20 nodes
experiments = 100
node = 20
epochs = 300

accs = []
for ex in range(experiments):
    accs.append(experiment(data, node, epochs))
    print("Nodes = "+str(node)+ " Experiment="+str(ex+1)+"/"+str(experiments))
accuracy = np.mean(accs)
error = np.std(accs)

#%% Two layer model
def model(hidden_nodes):
    return tf.keras.models.Sequential([
        tf.keras.layers.DenseFeatures(feature_columns=feature_columns),
        tf.keras.layers.Dense(units=hidden_nodes, activation='relu'),
        tf.keras.layers.Dropout(rate = 0.2),
        tf.keras.layers.Dense(units=hidden_nodes, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])

mod = model(64)
mod.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# Train it
history = mod.fit(train, validation_data=test, epochs=300)

# Measure metrics
acc = mod.evaluate(test)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'])

#%% Plot the histogram of results
bins = np.linspace(0, 1, 51)
plt.hist(accs, bins, density=True)
plt.xlabel('Accuracy')
plt.ylabel('Proability Density')
plt.plot([np.mean(accs), np.mean(accs)], [0, 14], linestyle=':', color='red')
plt.xlim([0.6, 1.0])

#%% Do a bootstrapped analysis of the accuracys
num = 10000
sz = np.size(accs)
means = []
for ex in range(num):
    nums = np.random.randint(sz, size=sz)
    newAccs = []
    for num in nums:
        newAccs.append(accs[num])
    means.append(np.mean(newAccs))

bins = np.linspace(0, 1, 51)
plt.hist(newAccs, bins, density=True)
plt.xlabel('Mean Bootstrapped Accuracy')
plt.ylabel('Proability Density')
plt.plot([np.mean(accs), np.mean(accs)], [0, 14], linestyle=':', color='red')
plt.xlim([0.6, 1.0])
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# Load the file
fileName = "heart.csv"
data = pd.read_csv(fileName)
X = data.drop(["target"], axis=1)
Y = data["target"]

#%%
# Correlation matrix
correl = data.corr()
plt.figure(figsize=(16, 6))
mask = np.triu(np.ones_like(correl, dtype=np.bool)) # Creates an upper-triangle matrix
sb.heatmap(correl, mask=mask, vmin=-0.5, vmax=0.5, annot=True, cmap='viridis')
plt.title('Correlation Heatmap')
plt.savefig('images/analysis/heatmap.png', dpi=300)

#%% Age and Sex data
ageData = data[['age', 'sex', 'target']]
bins = np.linspace(ageData['age'].min(), ageData['age'].max(), 9)
bins[0] -= 1


femaleData = ageData[ageData['sex'] == 0]
maleData = ageData[ageData['sex'] == 1]
sexNumbers = [[femaleData[femaleData['target']==1].count()[0],
               femaleData.count()[0] - femaleData[femaleData['target']==1].count()[0]],
              [maleData[maleData['target']==1].count()[0],
              maleData.count()[0] - maleData[maleData['target']==1].count()[0]]]
sexNumbers = np.array(sexNumbers)
plt.figure()
plt.bar(['Female', 'Male'], sexNumbers[:, 0], label='Heart Disease', width=0.6)
plt.bar(['Female', 'Male'], sexNumbers[:, 1], bottom=sexNumbers[:, 0], label='No Heart Disease', width=0.6)
plt.legend()
plt.savefig('images/analysis/male vs female.png', dpi=300)

#%% Principle Component Analysis
model = PCA()
X = data.drop(["target"], axis=1)
Y = data["target"]
model.fit(X)
var = model.explained_variance_ratio_
new_X = model.fit_transform(X)
plt.figure()
plt.scatter(new_X[np.where(Y == 0), 0], new_X[np.where(Y == 0), 1], color='blue')
plt.scatter(new_X[np.where(Y == 1), 0], new_X[np.where(Y == 1), 1], color='red')
plt.xlabel("PCA 1 ("+str(round(var[0]*100, 1))+"%)")
plt.ylabel("PCA 2 ("+str(round(var[1]*100, 1))+"%)")
plt.title("Principle Component Analysis")
plt.legend(["No heart disease", "Heart Disease"])

plt.figure()
plt.bar(np.linspace(1, np.size(var), np.size(var)), var)
plt.xlim([0.5, 5.5])
plt.ylim([0, 1])
plt.xlabel('PCA Axis')
plt.ylabel('Variance %')
plt.title("Scree Plot")


#%% Age vs thalach
plt.figure()
hd = data.loc[data['target'] == 1]
nhd = data.loc[data['target'] == 0]

plt.figure()
plt.scatter(hd['age'], hd['thalach'], color='blue')
plt.scatter(nhd['age'], nhd['thalach'], color='red')
plt.xlabel('Age')
plt.ylabel('Maximum recorded heart rate')
plt.legend(['Heart disease', 'No heart disease'])

#%% cp vs exang
nums = []
for cp in [0, 1, 2, 3]:
    nums.append([data.loc[(data['cp']==cp) & (data['exang']==0)].count()[0],
                 data.loc[(data['cp']==cp) & (data['exang']==1)].count()[0]])
nums = np.asarray(nums)

plt.bar(['0', '1', '2', '3'], nums[:, 0], label='exang=0', width=0.6)
plt.bar(['0', '1', '2', '3'], nums[:, 1], bottom=nums[:, 0], label='exang=1', width=0.6)
plt.legend()
plt.xlabel('Chest pain type')
plt.ylabel('Number')
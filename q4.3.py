import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA

data = pd.read_csv('iris.data', header=None)
data = data.sample(frac=1).reset_index(drop=True)
data[4] = data[4].replace({"Iris-virginica": 0, "Iris-versicolor": 1, "Iris-setosa": 2})
X = data.loc[:, 0:3].to_numpy()
y = data.loc[:, 4].to_numpy()

for col in range(X.shape[1]):
    X[:, col] = (X[:, col] - X[:, col].mean()) / X[:, col].std()

transformer = KernelPCA(n_components=2, kernel='sigmoid')
Z = transformer.fit_transform(X)

plt.figure()
plt.scatter(Z[:, 0], Z[:, 1], c=y)
plt.show()

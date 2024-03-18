import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets 
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix,classification_report


dataset = datasets.load_iris()

X = scale(dataset.data)
y = pd.DataFrame(dataset.target)
variable_names = dataset.feature_names

clustering = KMeans(n_clusters= 3,random_state=5)

clustering.fit(X)

iris_df = pd.DataFrame(dataset.data)
iris_df.columns = ['Sepal_length','Sepal_width','Petal_length','Petal_width']
y.columns = ['Target']

color_theme = np.array(['red','green','blue'])

relable = np.choose(clustering.labels_,[0,2,1]).astype(np.int64)

plt.subplot(1,2,1)
plt.scatter(x= iris_df['Petal_length'] , y = iris_df['Petal_width'],c = color_theme[dataset.target],s=50)

plt.subplot(1,2,2)
plt.scatter(x= iris_df['Petal_length'] , y = iris_df['Petal_width'],c = color_theme[relable],s=50)

plt.show()


print(classification_report(y,relable))
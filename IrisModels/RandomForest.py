import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

df = pd.DataFrame(iris.data, columns = iris.feature_names)
df['Species'] = pd.Categorical.from_codes(iris.target,iris.target_names)

X = df.iloc[:,0:4]
y = df.iloc[:,4]

X_train , X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2,random_state=0)

forest = RandomForestClassifier(n_jobs=2,random_state= 0)

forest.fit(X_train,y_train)

y_pred = forest.predict(X_test)

print(accuracy_score(y_test,y_pred))
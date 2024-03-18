import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dataset = pd.read_csv("Datasets/insurance2.csv")


X = dataset.iloc[:,0:5]
y = dataset.iloc[:,5]

X_train, X_test,y_train,y_test = train_test_split(X , y , test_size=0.2, random_state = 0)

decision = DecisionTreeClassifier(criterion= 'entropy',random_state= 100)

decision.fit(X_train,y_train)

y_pred = decision.predict(X_test)

print(accuracy_score(y_test,y_pred))
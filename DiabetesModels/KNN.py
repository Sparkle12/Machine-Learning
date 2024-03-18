import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


dataset = pd.read_csv('Datasets/diabetes.csv')

zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']

for column in zero_not_accepted:
    dataset[column] = dataset[column].replace(0, np.NaN)
    mean = int(dataset[column].mean(skipna=True))
    dataset[column] = dataset[column].replace(np.NaN, mean)

X = dataset.iloc[:, 1:8].drop(["SkinThickness", "BloodPressure"],axis=1)
y = dataset.iloc[:, 8 ]
X_train , X_test , y_train, y_test = train_test_split(X,y,test_size=0.2,random_state= 0)

print(X.head())

ss = StandardScaler()

X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

knn = KNeighborsClassifier(n_neighbors= 17, p = 2,metric="euclidean")

knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test,y_pred)

print(cm)

print(accuracy_score(y_test,y_pred))

print(f1_score(y_test,y_pred))

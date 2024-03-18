import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv("Datasets/diabetes.csv")

zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']

for column in zero_not_accepted:
    dataset[column] = dataset[column].replace(0, np.NaN)
    mean = int(dataset[column].mean(skipna=True))
    dataset[column] = dataset[column].replace(np.NaN, mean)


X = dataset.iloc[:, 0 : 8].drop(['SkinThickness','BloodPressure'],axis=1)
y = dataset.iloc[:,8]

X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state= 0)

ss = StandardScaler()

X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)

lr = LogisticRegression(solver="liblinear")

lr.fit(X_train_scaled,y_train)

y_pred = lr.predict(X_test_scaled)

cm = confusion_matrix(y_test,y_pred)
print(cm)

print(accuracy_score(y_test,y_pred))


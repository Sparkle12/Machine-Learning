import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


dataset = pd.read_csv("Datasets/insurance.csv")

dataset["sex"] =  dataset["sex"].astype("category")
dataset["sex"] = dataset["sex"].cat.codes

dataset["smoker"] =  dataset["smoker"].astype("category")
dataset["smoker"] = dataset["smoker"].cat.codes

dataset["region"] =  dataset["region"].astype("category")
dataset["region"] = dataset["region"].cat.codes

X = dataset.iloc[:,0:6].drop(["sex","children","region"],axis=1)
y = dataset.iloc[:,6]

X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

ss = StandardScaler()

X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)


lr = LinearRegression()

lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)


print(mean_squared_error(y_test,y_pred))
print(r2_score(y_true=y_test,y_pred = y_pred))




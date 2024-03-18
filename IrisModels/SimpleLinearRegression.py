import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

dataset = sns.load_dataset('iris')

X = dataset.iloc[:,2:3]
y = dataset.iloc[:,3:4]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state= 0)

lr = LinearRegression()

lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)

print(mean_squared_error(y_test,y_pred))




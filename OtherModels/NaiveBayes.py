import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


dataset = pd.read_csv("Datasets/discount.csv")

dataset['Day'] = dataset['Day'].astype('category')
dataset['Day'] = dataset['Day'].cat.codes

dataset['Discount'] = dataset['Discount'].astype('category')
dataset['Discount'] = dataset['Discount'].cat.codes

dataset['Free Delivery'] = dataset['Free Delivery'].astype('category')
dataset['Free Delivery'] = dataset['Free Delivery'].cat.codes

dataset['Purchase'] = dataset['Purchase'].astype('category')
dataset['Purchase'] = dataset['Purchase'].cat.codes

X = dataset.iloc[:,0:3]
y = dataset.iloc[:,3]

X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

model = MultinomialNB()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print(accuracy_score(y_test,y_pred))
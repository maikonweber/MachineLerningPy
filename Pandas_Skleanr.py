import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#print(train.head())
#print(test.head())

from sklearn.neighbors import KNeighborsClassifier #Classificador KNN

cols = [ 'shoe size', 'height']
cols2 = ['class']

  ,m   .x_train = train.to_numpy[cols]

y_train  = train.to_numpy(cols2)
x_test = test.to_numpy(cols)
y_test = test.to_numpy(cols2)

knn = KNeighborsClassifier(n_neighbors = 3 , weights = "distance" )
knn.fit(x_train, y_train.ravel())
output = knn.predict(x_test)
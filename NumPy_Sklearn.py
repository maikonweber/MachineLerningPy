import numpy as np
from sklearn import neighbors

# x são as entradas e y são as saídas;

x = np.genfromtxt('dataset2.data', delimiter=',', usecols=(1,2,3,4))
y = np.genfromtxt('dataset2.data', delimiter=',', usecols=(0))

#print(len(x))
#print(len(y))
#print(x)
#print(y)

from sklearn.model_selection import train_test_split #Retirar modelos de teste 



x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size= 0.3 , random_state=42)

#print(len(x_treino))
#print(x_treino)
#print(len(x_teste))


from sklearn.neighbors import KNeighborsClassifier # Fazer as Classficações.

knn = KNeighborsClassifier(n_neighbors=17, p=2)
knn.fit(x_treino, y_treino)
labels = knn.predict(x_teste)
print(len(labels))


print(np.sum(labels == y_teste))
print( 100 * (labels == y_teste).sum() / len(x_teste))

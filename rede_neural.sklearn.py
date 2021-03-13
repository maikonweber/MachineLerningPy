from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

print(cancer.feature_names)
print(cancer.target_names)
print(len(cancer.data))
print(cancer.keys())
print(cancer.DESCR)


X, y = cancer["data"], cancer['target']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)

print(len(X_train))
print(len(X_test))


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(30, 30, 30))
mlp.fit(X_train, y_train)

MLPClassifier(activation="relu", alpha=0.0001, batch_size="auto", beta_1=0.9,
beta_2=0.999, early_stopping=False, epsilon=1e-08, hidden_layer_sizes=(30, 30, 30), learning_rate='constant',
learning_rate_init=0.001, max_iter=200, momentum=0.9, nesterovs_momentum=True, power_t=0.5, random_state=None,
shuffle=True, solver="adam", tol=0.0001, validation_fraction=0.1,
verbose=False, warm_start=False)

predictions = mlp.predict(X_test)

print("S: ", mlp.score(X_test, y_test))

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, predictions))

print(classification_report(y_test, predictions))
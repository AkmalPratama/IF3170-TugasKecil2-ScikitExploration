from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=4)

# kNN
### Full training
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X, y)
y_pred = knn.predict(X)

acc = metrics.accuracy_score(y, y_pred)
print('Training result %f' % (acc))

#### Split training
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

acc = metrics.accuracy_score(y_test, y_pred)
print('Training result %f' % (acc))

### 10-fold

# MLP
### Full training
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(10),solver='sgd',learning_rate_init=0.01,max_iter=500)
mlp.fit(X, y)

print(mlp.score(X_test,y_test))

### Split training
mlp = MLPClassifier(hidden_layer_sizes=(10),solver='sgd',learning_rate_init=0.01,max_iter=500)
mlp.fit(X_train, y_train)

print(mlp.score(X_test,y_test))
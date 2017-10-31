import numpy as np

from sklearn import datasets, neighbors
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.9, test_size=0.1, random_state=4)

### kNN full training
print("\n")
print("kNN full training")
print("=================")
knn = KNeighborsClassifier()
y_pred = knn.fit(X, y).predict(X)
print("Number of mislabeled points out of a total %d points : %d" % (iris.data.shape[0], (iris.target != y_pred).sum()))

#### kNN 90% train, 10% test
print("\n")
print("kNN 90% train, 10% test")
print("=======================")
knn = KNeighborsClassifier()
y_pred = knn.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d" % (len(X_test), (y_test != y_pred).sum()))
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
plt.show()


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
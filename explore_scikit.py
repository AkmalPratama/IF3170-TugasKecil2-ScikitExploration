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


### kNN 10-fold cross validation
print("\n")
print("kNN 90% train, 10% test")
print("=======================")
kf = KFold(n_splits = 10, shuffle = False)
print(kf)
i = 1
temp = 0
for train_index, test_index in kf.split(iris.data):
    print("Fold ", i)
    print("TRAIN :", train_index, "\nTEST :", test_index)
    x_train = iris.data[train_index]
    x_test = iris.data[test_index]
    y_train = iris.target[train_index]
    y_test = iris.target[test_index]
    i += 1
    y_pred = knn.fit(x_train, y_train).predict(x_test)
    print("Number of mislabeled points out of a total %d points : %d" % (len(x_test), (y_test != y_pred).sum()))
    temp += (y_test != y_pred).sum()
print("Sum of mislabeled points : %d" % temp)
print("Mean of mislabeled points : %.4f" % float(temp/10))

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
import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
classifier = svm.SVC(kernel='linear', C=0.01)
y_pred = classifier.fit(X_train, y_train).predict(X_test)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




'''
### read tennis.csv dataset
import pandas as pd
tennis = pd.read_csv("D:/Kuliah/Semester 5/AI/Tucil 2/tennis.csv")
print(tennis)
'''





### NaiveBayes full training
print("NaiveBayes full training")
print("========================")
from sklearn import datasets
iris = datasets.load_iris()
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
print("Number of mislabeled points out of a total %d points : %d" % (iris.data.shape[0], (iris.target != y_pred).sum()))

### NaiveBayes 90% train, 10% test
print("\n")
print("NaiveBayes 90% train, 10% test")
print("==============================")
iris_data_train, iris_data_test, iris_target_train, iris_target_test = train_test_split(iris.data, iris.target, train_size = 0.9, test_size = 0.1)
y_pred_train = gnb.fit(iris_data_train, iris_target_train).predict(iris_data_test)
print("Number of mislabeled points out of a total %d points : %d" % (len(iris_data_test), (iris_target_test != y_pred_train).sum()))
# Compute confusion matrix
cnf_matrix = confusion_matrix(iris_target_test, y_pred_train)
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

### NaiveBayes 10-fold cross validation
print("\n")
print("NaiveBayes 10-fold cross validation")
print("===================================")
from sklearn.model_selection import KFold
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
    y_pred = gnb.fit(x_train, y_train).predict(x_test)
    print("Number of mislabeled points out of a total %d points : %d" % (len(x_test), (y_test != y_pred).sum()))
    temp += (y_test != y_pred).sum()
print("Sum of mislabeled points : %d" % temp)
print("Mean of mislabeled points : %.4f" % float(temp/10))





### DecisionTree full training
print("\n")
print("DecisionTree full training")
print("==========================")
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)
import graphviz 
dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)  
graph
print("Number of mislabeled points out of a total %d points : %d" % (iris.data.shape[0], (iris.target != clf.predict(iris.data[:, :])).sum()))

### DecisionTree 90% train, 10% test
print("\n")
print("DecisionTree 90% train, 10% test")
print("================================")
iris_data_train, iris_data_test, iris_target_train, iris_target_test = train_test_split(iris.data, iris.target, train_size = 0.9, test_size = 0.1)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris_data_train, iris_target_train)
dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)  
graph
print("Number of mislabeled points out of a total %d points : %d" % (len(iris_data_test), (iris_target_test != clf.predict(iris_data_test[:, :])).sum()))
# Compute confusion matrix
cnf_matrix = confusion_matrix(iris_target_test, clf.predict(iris_data_test[:, :]))
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

### DecisionTree 10-fold cross validation
print("\n")
print("DecisionTree 10-fold cross validation")
print("=====================================")
kf = KFold(n_splits = 10, shuffle = False)
print(kf)
i = 1
temp = 0
clf = tree.DecisionTreeClassifier()
for train_index, test_index in kf.split(iris.data):
    print("Fold ", i)
    print("TRAIN :", train_index, "\nTEST :", test_index)
    x_train = iris.data[train_index]
    x_test = iris.data[test_index]
    y_train = iris.target[train_index]
    y_test = iris.target[test_index]
    i += 1
    clf = clf.fit(x_train, y_train)
    dot_data = tree.export_graphviz(clf, out_file=None,
                             feature_names=iris.feature_names,  
                             class_names=iris.target_names,  
                             filled=True, rounded=True,  
                             special_characters=True)  
    graph = graphviz.Source(dot_data)  
    graph
    print("Number of mislabeled points out of a total %d points : %d" % (len(x_test), (y_test != clf.predict(x_test[:, :])).sum()))
    temp += (iris.target != clf.predict(iris.data[:, :])).sum()
print("Sum of mislabeled points : %d" % temp)
print("Mean of mislabeled points : %.4f" % float(temp/10))
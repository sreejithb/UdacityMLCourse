from sklearn.metrics import accuracy_score

def check_accuracy(clf,X_test, labels):
    Z = clf.predict(X_test);
    return accuracy_score(labels,Z)

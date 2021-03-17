# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 09:36:17 2021

@author: user
"""

from sklearn import svm, datasets #menginport class dataset dari scikit learn library
clf = svm.SVC(gamma=0.001, C=100.) #memanggil class SVC dan menset argumen constructor SVC serta ditampung di variabel clf
X, y = datasets.load_iris(return_X_y=True) #meload datasets iris dan ditampung di variabel x untuk data dan y untuk target
clf.fit(X, y) #memanggil method fit untuk melakukan training data dengan argumen data dan target dari datasets iris


#Pickle
import pickle
s = pickle.dumps(clf)
clf2 =pickle.loads(s)
print(clf.predict(X[0:1]))

#Joblib
from joblib import dump, load
dump(clf, '1184038.joblib')
clf3 = load('1184038.joblib')
print(clf3.predict(X[0:1]))
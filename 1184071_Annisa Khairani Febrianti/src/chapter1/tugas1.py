# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 19:06:59 2021

@author: ACER
"""
#%% 1 Loading an example dataset

from sklearn import datasets 
#digunakan untuk memanggil class dataset
iris = datasets.load_iris()
#menggunakan contoh datasets iris
digits = datasets.load_digits()
#menyimpan nilai data sets iris
print(digits.target)
#menampilkan hasil dari variabel digits
#%% Learning and Predicting

from sklearn import svm 
#untuk mengimport class svm dari library sklearn
from sklearn import datasets
#untuk mengimport class/fungsi data sets
clf = svm.SVC(gamma=0.001, C=100.)
#memasukkan  implementasi dari "Support Vector Classification" ke variable clf
iris = datasets.load_iris()
#untuk memuat dan memasukkan dataset iris ke variable bernama iris
digits = datasets.load_digits() 
#untuk memuat dan memasukkan data set digit ke varable digits
clf.fit(digits.data[:-1], digits.target[:-1]) 
#untuk melakukan pengiriman data training set ke method fit
clf.predict(digits.data[-1:]) 
#untuk melakukan prediksi nilai pada digits


#%% Model Persistences
from sklearn import svm 
#digunakan untuk memanggil svn dilibrary
from sklearn import datasets 
#digunakan untuk memanggil svn dilibrary
clf = svm.SVC()
#memberikan nilai gama secara manual
X, y = datasets.load_iris(return_X_y=True)
#mengambil data sets iris
clf.fit(X, y) 
#clf sebagai classfilter
#%%
import pickle 
#mengambil library pickle
s = pickle.dumps(clf)
#unuk membuat variable s sebagai classfier
clf2 = pickle.loads(s) 
#variable clf2 sebagai load(s)
clf2.predict(X[0:1]) 
#untuk prediksi
#%%
from joblib import dump, load
#mengambil dump,melaui library joblib
dump(clf, '1184006.joblib') 
#menyimpan model kedalam 1174027.joblib
clf = load('1184006.joblib') 
#memanggil model 1184006
print(clf) 
#menampilkan hasil model clf
#%%  Convetion
import numpy as np 
#memanggil library numpy dan dibuat alias np
from sklearn import random_projection
#memanggil class random_project

rng = np.random.RandomState(0) 
#membuat variable rng, mendefiniskan np,fungsi random dan attr Random
X = rng.rand(10, 2000) #untuk membuat variable X,dan menetukan nilai X
X = np.array(X, dtype='float32')
#untukmenyiman nilai random dengan tife data float32
X.dtype #mengubah tife data menjadi float64
 
transformer = random_projection.GaussianRandomProjection() 
#membuat variable tranformer dan mendefinisikan random_projection
X_new = transformer.fit_transform(X)
#membuat variable x yang baru
X_new.dtype #mengubah tife data menjadi 64
print(X_new) #menampilkan hasil
#%%  Error
from sklearn import svm 
#untuk mengimport class svm dari library sklearn
clf = svm.SVC(gamma=0.001, C=100.)
#memasukkan  implementasi dari "Support Vector Classification" ke variable clf
clf.fit(digits.data[:-1], digits.target[:-1]) 
#untuk melakukan pengiriman data training set ke method fit
clf.predict(digits.data[-1:]) 
#untuk melakukan prediksi nilai pada digits




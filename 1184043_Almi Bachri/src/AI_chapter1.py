# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 19:50:55 2021

@author: lovo
"""
#a Loading an example dataset
from sklearn import datasets #Digunakan untuk memanggil class datasets dari library sklearn
iris = datasets.load_iris() #Menggunakan contoh datasets iris
digits = datasets.load_digits() #Menyimpan nilai datasets iris pada variabel digits

print(digits.target) #Menampilkan hasil dari variabel digits

# Learning and predicting
from sklearn import svm #Digunakan untuk memanggil svm pada library sklearn
clf = svm.SVC(gamma=0.001, C=100.) #Memberikan nilai gama secara manual
clf.fit(digits.data[:-1], digits.target[:-1]) #clf sebagai classifier dan kemudian set training dengan metode fit
clf.predict(digits.data[-1:]) #Memprediksi nilai baru dari digits data 

#Model persistence
from sklearn import svm #Digunakan untuk memanggil svm pada library sklearn
from sklearn import datasets #Digunakan untuk memanggil class datasets dari library sklearn
clf = svm.SVC() # Membuat varibael cif, dan memanggil class svm dari fungsi SVC
X, y = datasets.load_iris(return_X_y=True) #Mengambil dataset iris dan mengembalikan nilainya
clf.fit(X, y) #Perhitungan nilai tabel


# Conventions
import numpy as np #Membuat library numpy dan dibuat alias  np
from sklearn import random_projection #Memanggil class random projection pada library sklearn

rng = np.random.RandomState(0) #Membuat variabel rng, dan mendefinisikan np, fungsi random dan attr RandomState kedalam variabel
X = rng.rand(10, 2000) #Membuat variabel x, dan menentukan nilai random dari (10-2000)
X = np.array(X, dtype='float32') #Utuk menyimpan hasil nilai random sebelumnya kedalam array, dan menentukan typedatanya sebagai float32
X.dtype #Mengubah data type menjadi float64


transformer = random_projection.GaussianRandomProjection() #Membuat variabel tranformer, dan mendefinisikan class random projection dan memanggil fungsi GausianRandomProjection
X_new = transformer.fit_transform(X) #Membuat varibael baru dan melakukan perhitungan label pada varibael x
X_new.dtype #Mengubah data type menjadi float64

print(X_new) #Menampilkan hasil

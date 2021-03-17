# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 09:11:46 2021

@author: user
"""

from sklearn import datasets #mengimport class dataset dari scikit learn library
iris = datasets.load_iris() #memuat dan memasukkan dataset iris ke variabel bernama iris
digits = datasets.load_digits() #memuat dan memasukkan dataset digits ke variabel digits

print(digits.data) #memberikan akses ke fitur yang dapat digunakan untuk mengklasifikasikan sample digit dan menampilkan di console
digits.target #memberikan informasi tentang data yang berhubungan atau juga dapat dijadikan sebagai label
digits.images[0] #Data selelu berupa array 2D, shape(n.samples, n.features), meskipun data aslinya mungkin memiliki bentuk yang berbeda

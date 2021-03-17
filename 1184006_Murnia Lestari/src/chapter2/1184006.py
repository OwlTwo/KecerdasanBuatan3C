# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 06:31:38 2021

@author: ACER
"""

#%% Mengganti Nama Variabel
print(1184006 % 3)
# mencari modulus dari hasil pembagian 3 dan menampilkan hasilnya
#%% Soal 1
# load dataset (student mat pakenya)
import pandas as pd
# mengimport lib pandas lalu dialiaskan menjadi pd
apel = pd.read_csv('student-mat.csv', sep=';')
# membaca file csv dengan method read_csv dari lib pandas dimana argumen 
#pertama nama file csv dan argumen kedua pemisah tiap datanya, 
#lalu ditampung pada variable apel
len(apel) 
# mengetahui jumlah data pada list
#%% Soal 2
# generate binary label (pass/fail) based on G1+G2+G3
# (test grades, each 0-20 pts); threshold for passing is sum>=30
apel['pass'] = apel.apply(lambda row: 1 if (row['G1']+row['G2']+row['G3']) >= 35 else 0, axis=1) 
# memanggil method apply dari lib pandas untuk menerapkan function yang dibuat dengan lambda dimana parameternya adalah row, 
# kemudian di cek apakah penjumlahan dari row['G1'], row['G2'], dan row['G3'] lebih dari sama dengan 35 maka akan mengembalikan
# 1 jika tidak maka mengembalikan 0 pada kolom pass, serta diterapkan di semua baris data
apel = apel.drop(['G1', 'G2', 'G3'], axis=1) 
# menghapus kolom G1, G2, dan G3
apel.head() 
# menampilkan lima baris pertama datanya
print(apel)
# mencetak variable
#%% Soal 3
# use one-hot encoding on categorical columns
apel = pd.get_dummies(apel, columns=['sex','school','address','famsize','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic']) 
# memanggil method get_dummies untuk membuat variabel dummy dari data apel dimana variable dummy yang dibuat adalah kolom 'sex','school','address','famsize','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup','paid','activities','nursery','higher','internet', dan 'romantic'
apel.head()
# menampilkan lima baris pertama datanya
print(apel)
# mencetak variable
#%% Soal 4
# shuffle rows
apel = apel.sample(frac=1) 
# memanggil method sample untuk membuat random sample dimana argumen fracnya 1 yg berarti semua baris datanya untuk diacak
# split training and testing data
apel_train = apel[:300] 
# menampung 300 data pertama yang ditampung di variabel apel_train
apel_test = apel[300:]
 # menampung sisa dari 300 data pertama yang ditampung di variabel apel_test

apel_train_att = apel_train.drop(['pass'], axis=1) 
# menghapus kolom pass pada variabel apel_train
apel_train_pass = apel_train['pass'] 
# menampung isi kolom pass pada variabel apel_test di variabel apel_train _pass

apel_test_att = apel_test.drop(['pass'], axis=1) 
# menghapus kolom pass pada variabel apel_test
apel_test_pass = apel_test['pass'] 
# menampung isi kolom pass pada variabel apel_test di variabel apel_test _pass

apel_att = apel.drop(['pass'], axis=1) 
# menghapus kolom pass pada variabel apel
apel_pass = apel['pass'] # menampung isi kolom pass di variabel apel

# number of passing students in whole dataset:
import numpy as np 
# mengimport lib numpy lalu dialiaskan sebagai np
print("Passing: %d out of %d (%.2f%%)" % (np.sum(apel_pass), len(apel_pass),100*float(np.sum(apel_pass))/len(apel_pass)))
 # menampilkan data passing

#%% Soal 5
# fit a decision tree
from sklearn import tree
 # mengimport lib tree dari sklearn
fuji = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5) 
# memanggil class DecisionTreeClassifier untuk mengklasifikasikan dimana constructornya memilik argumen criterion = entropy dan argumen max_depth = 5
fuji = fuji.fit(apel_train_att, apel_train_pass) 
# memanggil method fit untuk dilakukan training dengan argumen pertama data/fiturnya dan argumen kedua label/targetnya
print (fuji)
# mencetak variabel
#%% Soal 6
# visualize tree
import graphviz
 # mengimport lib graphviz
dot_data = tree.export_graphviz(fuji, out_file=None, label="all",impurity=False, proportion=True,feature_names=list(apel_train_att),class_names=["fail","pass"],filled=True, rounded=True) 
# memanggil method export_graphiz untuk mengekspor decision tree ke format DOT dan ditampung di variabel dot_data dengan argumen pertama decision tree yang di ekspor, argumen kedua nama file DOT dibiarkan default, argumen ketiga untuk menampilkan semua labelnya, argumen keempat impurity tidak ditampilkan, argumen kelima proportion ditampilkan presentase samplenya, argumen keenam nama fiturnya sesuai dengan variabel apel_train_att, argumen ketujuh nama class/target adalah fail dan pass, argumen kedelapan filled adalah True untuk menujukkan tipenya, dan argumen kesembilan tampilan sudut kotaknya lingkaran
merah = graphviz.Source(dot_data) 
# memanggil Source dengan argumen contructor sumber data DOT dan ditampung di variabel merah
merah
print (merah)
# mencetak variabel
#%% Soal 7
# save tree
tree.export_graphviz(fuji, out_file="student-performance.dot",label="all", impurity=False,proportion=True,feature_names=list(apel_train_att),class_names=["fail", "pass"],filled=True, rounded=True)
# memanggil method export_graphiz untuk mengekspor decision tree ke format DOT dengan argumen pertama decision tree yang di ekspor, argumen kedua nama file DOT, argumen ketiga untuk menampilkan semua labelnya, argumen keempat impurity tidak ditampilkan, argumen kelima proportion ditampilkan presentase samplenya, argumen keenam nama fiturnya sesuai dengan variabel apel_train_att, argumen ketujuh nama class/target adalah fail dan pass, argumen kedelapan filled adalah True untuk menujukkan tipenya, dan argumen kesembilan tampilan sudut kotaknya lingkaran

#%% Soal 8
fuji.score(apel_test_att, apel_test_pass) 
# memanggil method score untuk menghitung rata-rata akurasi
#%% Soal 9
from sklearn.model_selection import cross_val_score
# mengimport cross_val_score
hijau = cross_val_score(fuji, apel_att, apel_pass, cv=5) 
# memanggil method cross_val_score untuk mengevaluasi kinerja model dengan dengan metode Kfolds dengan argumen pertama objek training, argumen kedua data/fitur training, argumen ketiga target training, dan argumen keempat cross validation bernilai 5
# show average score and +/- two standard deviations away
#(covering 95% of scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (hijau.mean(), hijau.std() * 2)) 
# menampilkan akurasi data

#%% Soal 10
for max_depth in range(1, 20): 
    # melakukan perulangan dalam rentang 1 sampai 20
    fuji = tree.DecisionTreeClassifier(criterion="entropy",max_depth=max_depth) 
    # memanggil class DecisionTreeClassifier untuk mengklasifikasikan dimana constructornya memilik argumen criterion = entropy dan argumen max_depth = 5
    hijau = cross_val_score(fuji, apel_att, apel_pass, cv=5) 
    # memanggil method cross_val_score untuk mengevaluasi kinerja model dengan dengan metode Kfolds dengan argumen pertama objek training, argumen kedua data/fitur training, argumen ketiga target training, dan argumen keempat cross validation bernilai 5
    print("Max depth: %d, Accuracy: %0.2f (+/- %0.2f)" %(max_depth, hijau.mean(), hijau.std() * 2)) 
    # menampilkan max depth dan akurasi data

#%% Soal 11
depth_acc = np.empty((19,3), float) 
# membuat array dengan nilai random bertipe float
i = 0 
# inisialisasi i
for max_depth in range(1, 20): 
    # melakukan perulangan dalam rentang 1 sampai 20
    hijau = tree.DecisionTreeClassifier(criterion="entropy",max_depth=max_depth) 
    # memanggil class DecisionTreeClassifier untuk mengklasifikasikan dimana constructornya memilik argumen criterion = entropy dan argumen max_depth = 5
    hijau = cross_val_score(fuji, apel_att, apel_pass, cv=5) 
    # memanggil method cross_val_score untuk mengevaluasi kinerja model dengan dengan metode Kfolds dengan argumen pertama objek training, argumen kedua data/fitur training, argumen ketiga target training, dan argumen keempat cross validation bernilai 5
    depth_acc[i,0] = max_depth 
    # menampung max_depth
    depth_acc[i,1] = hijau.mean() 
    # menampung rata-rata kinerja model
    depth_acc[i,2] = hijau.std() * 2 
    # emnampung standar deviasi 
    i += 1 
    # melakukan increase
    
depth_acc 
# Depth acc akan membuat array kosong dengan mengembalikan array baru dengan bentuk dan tipe yang diberikan
print(depth_acc) 
# Mencetak depth_acc
#%% Soal 12
import matplotlib.pyplot as plt 
# mengimport lib pyplot dialiaskan sebagai plt
fig, ax = plt.subplots() 
# memanggil method subplots untuk membuat subplot berjumlah 1 buah
ax.errorbar(depth_acc[:,0], depth_acc[:,1], yerr=depth_acc[:,2]) 
# dengan argumen pertama dan kedua lokasi dari data, argumen ketiga ukuran y dari errorbar
plt.show() 
# menampilkan plotnya
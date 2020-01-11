# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 17:07:16 2019

@author: Ali
"""

import pandas as pd
from sklearn.decomposition import KernelPCA
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.metrics as sm
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D

#Importing Dataset
Gene_Data = pd.read_csv('data_linear_CNA.csv')
Patient_Data = pd.read_csv('data_clinical_sample.csv')

#Extracting Y Value
Y = Patient_Data[['IHC_HER2', 'SAMPLE_ID']]
S = 'IHC_HER2'

#Drop ID Columns
Gene_Data = Gene_Data.drop('Hugo_Symbol',axis=1)
Gene_Data = Gene_Data.drop('Entrez_Gene_Id',axis=1)

#Transposing Data
X_old = Gene_Data.transpose()

#Normalizing Data to get rid of negative values
mm_scaler = preprocessing.MinMaxScaler()
X_old = pd.DataFrame(mm_scaler.fit_transform(X_old))
Y = Y.set_index('SAMPLE_ID')
X = pd.concat([X_old, pd.DataFrame(Y.values)], axis=1, ignore_index=True)
X = X.rename(columns={22247: "IHC_HER2"})
X = X.drop(X.index[-1])
X = X.drop(X.index[-1])

#Bar Chart to Display Classes
Y.IHC_HER2.value_counts().plot(kind='bar', title='Before Over Sampling');

#Replacing Classnames with integers
ClassList = []
for item in X[S]:
    if item == 'Negative':
        ClassList.append(0)
    elif item == 'Equivocal':
        ClassList.append(1)
    elif item == 'Positive':
        ClassList.append(2)
    else:
        ClassList.append(-1)

#Adding Y to dataset
X[S] = ClassList

#Seperating Classes
D0 = X.query('IHC_HER2 == "0"')
D1 = X.query('IHC_HER2 == "1"')
D2 = X.query('IHC_HER2 == "2"')

#Appending Selected Classes
DF = D0.append(D1).append(D2)

#Bar Chart to Display Selected Classes
DF.IHC_HER2.value_counts().plot(kind='bar', title='Selected');

X_o = DF.iloc[:, 0:22247]
Y_o = DF['IHC_HER2']

#OverSampling unbalanced classes
smo = SMOTE(random_state=42,k_neighbors=10, sampling_strategy='not majority')
X_s, Y_s = smo.fit_resample(X_o, Y_o)

#Making Dataframes to make Bar Chart
X_d = pd.DataFrame(X_s)
Y_d = pd.DataFrame(Y_s)
X_d['IHC_HER2'] = Y_d

#Bar Chart to Display Selected Classes after Oversampling
X_d.IHC_HER2.value_counts().plot(kind='bar', title='After Over Sampling');

#Selecting K-Best Features
X_new = SelectKBest(chi2, k=18000).fit_transform(X_s,Y_s)

#Splitting Data to Training and Testing Data
Points_train, Points_test, Label_train, Label_test = train_test_split(X_new, Y_s,train_size=0.3,shuffle=True)   

#Applying Support Vector Machine Classifier with rbf kernel
svm = SVC(gamma= 0.1, kernel='linear')
svm.fit(Points_train, Label_train)
Label_pred = svm.predict(Points_test)

#Checking SVM accuracy
print("SVM",str(round(sm.accuracy_score(Label_test, Label_pred)*100,1)))

#Reducing dimensions for data visualization
kpca = KernelPCA(n_components=3, kernel='rbf',gamma=0.1)
X_pca = kpca.fit_transform(X_new)

#Plotting Graph
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection='3d')
x = X_pca[:,0]
y = X_pca[:,1]
z = X_pca[:,2]
ax.scatter(x, y, z, c=Y_s,s=100, marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()








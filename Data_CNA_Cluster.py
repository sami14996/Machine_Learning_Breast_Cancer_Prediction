# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 20:40:35 2019

@author: Ali
"""
# -*- coding: utf-8 -*-

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
from sklearn.cluster import SpectralClustering
from sklearn import mixture
from sklearn.metrics import calinski_harabasz_score 

#Importing Dataset
Gene_Data = pd.read_csv('data_linear_CNA.csv')
Patient_Data = pd.read_csv('data_clinical_sample.csv')

#Extracting Y Value
Y = Patient_Data[['ER_STATUS_BY_IHC', 'SAMPLE_ID']]
S = 'ER_STATUS_BY_IHC'

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
X = X.rename(columns={22247: "ER_STATUS_BY_IHC"})
X = X.drop(X.index[-1])
X = X.drop(X.index[-1])

#Bar Chart to Display Classes
Y.ER_STATUS_BY_IHC.value_counts().plot(kind='bar', title='Before Over Sampling');

#Replacing Classnames with integers
ClassList = []
for item in X[S]:
    if item == 'Negative':
        ClassList.append(0)
    elif item == 'Positive':
        ClassList.append(1)
    else:
        ClassList.append(-1)

#Adding Y to dataset
X[S] = ClassList

#Seperating Classes
D0 = X.query('ER_STATUS_BY_IHC == "0"')
D1 = X.query('ER_STATUS_BY_IHC == "1"')

#Appending Selected Classes
DF = D0.append(D1)

#Bar Chart to Display Selected Classes
DF.ER_STATUS_BY_IHC.value_counts().plot(kind='bar', title='Selected');

X_o = DF.iloc[:, 0:22248]
Y_o = DF['ER_STATUS_BY_IHC']

#OverSampling unbalanced classes
smo = SMOTE(random_state=42,k_neighbors=10, sampling_strategy='not majority')
X_s, Y_s = smo.fit_resample(X_o, Y_o)

#Making Dataframes to make Bar Chart
X_d = pd.DataFrame(X_s)
Y_d = pd.DataFrame(Y_s)
X_d['ER_STATUS_BY_IHC'] = Y_d


#Bar Chart to Display Selected Classes after Oversampling
X_d.ER_STATUS_BY_IHC.value_counts().plot(kind='bar', title='After Over Sampling');

#%%
#Selecting K-Best Features
X_new = SelectKBest(chi2, k=500).fit_transform(X_s,Y_s)

import seaborn as sns; sns.set()
plt.figure(figsize=(10, 20))
sns.heatmap(X_new, annot=False, fmt="f")

H = pd.DataFrame(X_new)
H['C'] = Y_s
X_new = H.iloc[:,:]

#kpca = KernelPCA(n_components=3, kernel='rbf',gamma=0.1)
#X_new = kpca.fit_transform(X_new)

#Applying Spectral Clustring
Label_pred = SpectralClustering(n_clusters=2,gamma=0.1,affinity="rbf").fit_predict(X_new)

#Reducing dimensions for data visualization
kpca = KernelPCA(n_components=3, kernel='rbf',gamma=0.1)
X_pca = kpca.fit_transform(X_new)

Pred_DF = pd.DataFrame(X_pca)

Pred_DF = Pred_DF.rename(columns={0: "x1"})
Pred_DF = Pred_DF.rename(columns={1: "x2"})
Pred_DF = Pred_DF.rename(columns={2: "x3"})

Pred_DF['Original'] = Y_d
Pred_DF['Predicted'] = Label_pred
#Seperation data Based on Label and Cluster
A = Pred_DF.loc[Pred_DF['Original']==0].loc[Pred_DF['Predicted']==0]
B = Pred_DF.loc[Pred_DF['Original']==0].loc[Pred_DF['Predicted']==1]
C = Pred_DF.loc[Pred_DF['Original']==1].loc[Pred_DF['Predicted']==0]
D = Pred_DF.loc[Pred_DF['Original']==1].loc[Pred_DF['Predicted']==1]

#Plotting 3D Graph
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(A.x1,A.x2,A.x3,s=100,c='lightgreen',marker='x',alpha = 0.5,label ='Class = 0 and Cluster = 0')
ax.scatter(B.x1,B.x2,B.x3,s=100,c='lightgreen',marker='v',alpha = 0.5,label ='Class = 0 and Cluster = 1')
ax.scatter(C.x1,C.x2,C.x3,s=100,c='orange',marker='x',alpha = 0.5,label ='Class = 1 and Cluster = 0')
ax.scatter(D.x1,D.x2,D.x3,s=100,c='orange',marker='v',alpha = 0.5,label ='Class = 1 and Cluster = 1')
ax.legend(loc='upper right')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

#%%
#Plotting 2D Graph
plt.figure(figsize=(8, 8))
plt.scatter(A.x1,A.x2,s=100,c='lightgreen',marker='x',alpha = 0.5,label ='Class = 0 and Cluster = 0')
plt.scatter(B.x1,B.x2,s=100,c='lightgreen',marker='v',alpha = 0.5,label ='Class = 0 and Cluster = 1')
plt.scatter(C.x1,C.x2,s=100,c='orange',marker='x',alpha = 0.5,label ='Class = 1 and Cluster = 0')
plt.scatter(D.x1,D.x2,s=100,c='orange',marker='v',alpha = 0.5,label ='Class = 1 and Cluster = 1')
plt.legend(loc='upper right')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Spectral Clustering on '+ S + '.csv')
plt.show()


#Reducing dimensions for data visualization
kpca = KernelPCA(n_components=3, kernel='rbf',gamma=0.1)
X_pca = kpca.fit_transform(X_s)

#Plotting Graph
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection='3d')
x = X_pca[:,0]
y = X_pca[:,1]
z = X_pca[:,2]
ax.scatter(x, y, z, c=Label_pred,s=100, marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()


#Plotting graph of values of CH-Index for different values of K
kx = []
ky = []
ex = []
ey = []
sx = []
sy = []

for k in range(2,12):
    km = KMeans(n_clusters=k, random_state=0)
    Label_pred = km.fit_predict(X_new)
    kx.append(k)
    ky.append(round(calinski_harabasz_score(X_new,Label_pred),4))
    
    em = mixture.GaussianMixture(n_components=k, covariance_type='full')
    em.fit(X_new)
    Label_pred = em.predict(X_new)
    ex.append(k)
    ey.append(round(calinski_harabasz_score(X_new,Label_pred),4))

plt.figure(figsize=(8, 8))
plt.legend(loc='upper right')
plt.xlabel('K')
plt.ylabel('CH-Index')
plt.plot(kx,ky,label = 'K-means',marker='v')
plt.plot(ex,ey,label = 'EM',marker='v')
#plt.plot(sx,sy,label = 'Spectral')
plt.title('CH-Index plot for values of K on '+ S)
plt.legend()
plt.show()



"""
#Splitting Data to Training and Testing Data
Points_train, Points_test, Label_train, Label_test = train_test_split(X_new, Y_s,train_size=0.5,shuffle=True)   

#Applying Support Vector Machine Classifier with rbf kernel
svm = SVC(gamma= 0.1, kernel='rbf')
svm.fit(Points_train, Label_train)
Label_pred = svm.predict(Points_test)

#Checking SVM accuracy
print("SVM",str(round(sm.accuracy_score(Label_test, Label_pred)*100,1)))


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







"""
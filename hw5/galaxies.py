#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 17:58:41 2021

@author: michail
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from forest import random_forest
import json


dt = pd.read_csv('sdss_redshift.csv')
xf = np.array(dt.loc[:, 'u':'z'])
y = np.array(dt['redshift'])

X_train, X_test, y_train, y_test = train_test_split(xf, y, train_size=0.75, random_state=42)

forest = random_forest(X_train, y_train, number=15)
y_pred_train = forest.predict(X_train)
y_pred_test = forest.predict(X_test)

plt.figure(figsize=(12,12))
plt.scatter(y_pred_train, y_train, label="Train", marker='*', s=2)
plt.scatter(y_pred_test, y_test, label='Test', alpha=0.3, s=2, color='red')
plt.xlabel('Real y', fontsize=18)
plt.ylabel('Test y', fontsize=18)
plt.legend(fontsize=18)
plt.savefig('redhift.png')

# save results
file = {"train": float('{:.3f}'.format(np.std((y_train - y_pred_train)))), "test": float('{:.3f}'.format(np.std((y_test - y_pred_test))))}
json_file = json.dumps(file) 
with open("redhsift.json", "w") as outfile: 
    outfile.write(json_file)
    
data = pd.read_csv('sdss.csv')
X = np.array(data)
Y = forest.predict(X)
data['redshift'] = Y
data.to_csv('sdss_predict.csv')

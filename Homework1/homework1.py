# -*- coding: utf-8 -*-
"""258_HW1

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/11uVPuRV_yXLSD4N3jMrnWPBe3vrAk0sR
"""

import json
from matplotlib import pyplot as plt
from collections import defaultdict
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import numpy
import random
import gzip
import math
import warnings
import numpy as np
warnings.filterwarnings("ignore")
import numpy as np

from sklearn.preprocessing import PolynomialFeatures

def assertFloat(x): # Checks that an answer is a float
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N
f = gzip.open("young_adult_10000.json.gz")
dataset = []
for l in f:
    dataset.append(json.loads(l))
#len(dataset)
answers = {}

dataset[0]
d = dataset
### Question 1
ratings=[]
lengths=[]
ratings = [d['rating'] for d in dataset]
lengths = [d['review_text'].count('!') for d in dataset]
X = numpy.asarray([[1,l] for l in lengths])
y = numpy.asarray(ratings).T
model = linear_model.LinearRegression(fit_intercept=False)
model.fit(X, y)
theta,residuals,rank,s = numpy.linalg.lstsq(X, y, rcond=None)
# theta
theta0 = theta[0]
theta1 = theta[1]
y_pred =  X @ theta

sse = sum([x**2 for x in (y - y_pred)])
mse = sse / len(y)
mse
mse = mean_squared_error(y, y_pred)
print(theta0,theta1,mse)
answers['Q1'] = [theta0, theta1, mse]
assertFloatList(answers['Q1'], 3)

### Question 2


X = np.asarray([[1, l, e] for l, e in zip([len(d['review_text']) for d in dataset],lengths)])
y = np.asarray(ratings).T
# model = linear_model.LinearRegression(fit_intercept=False)
# model.fit(X, Y)
# theta0 = model.intercept_
# theta1 = model.coef_[0]
# theta2 = model.coef_[1]
# y_pred = model.predict(X)
theta,residuals,rank,s = numpy.linalg.lstsq(X, y, rcond=None)
theta
theta0 = theta[0]
theta1 = theta[1]
theta2 = theta[2]
y_pred =  X @ theta

sse = sum([x**2 for x in (y - y_pred)])
mse = sse / len(y)
mse
print(theta0,theta1,theta2,mse)
answers['Q2'] = [theta0, theta1, theta2, mse]
assertFloatList(answers['Q2'], 4)

### Question 3

X = np.array(lengths).reshape(-1, 1)
y = np.array(ratings)

mse_r = {}

for deg in range(1, 6):
    poly = PolynomialFeatures(degree=deg)
    X_p = poly.fit_transform(X)
    model = linear_model.LinearRegression(fit_intercept=False)
    model.fit(X_p, y)
    mse_r[deg] = mean_squared_error(y,model.predict(X_p))


for deg, mse in mse_r.items():
    print(deg,mse)
answers['Q3'] = [mse_r[1],mse_r[2],mse_r[3],mse_r[4],mse_r[5]]
assertFloatList(answers['Q3'], 5)

### Question 4
X_train = np.asarray([[1, l] for l in lengths[:len(dataset) // 2]])
y_train = np.asarray(ratings[:len(dataset) // 2]).T
X_test = np.asarray([[1, l] for l in lengths[len(dataset) // 2:]])
y_test = np.asarray(ratings[len(dataset) // 2:]).T

mse_r2 = {}

for deg in range(1, 6):
    poly = PolynomialFeatures(degree=deg)
    X_p = poly.fit_transform(X_train)
    model = linear_model.LinearRegression(fit_intercept=False)
    model.fit(X_p, y_train)
    X_p2 = poly.fit_transform(X_test)
    mse_r2[deg] = mean_squared_error(y_test,model.predict(X_p2))

for deg, mse in mse_r2.items():
    print(deg,mse)

answers['Q4'] = [mse_r2[1],mse_r2[2],mse_r2[3],mse_r2[4],mse_r2[5]]
assertFloatList(answers['Q4'], 5)

### Question 5

mae = np.mean(np.abs(np.array(ratings[len(dataset) // 2:]) - np.mean(ratings[len(dataset) // 2:])))

print(np.mean(ratings[len(dataset) // 2:]), mae)
answers['Q5'] = [np.mean(ratings[len(dataset) // 2:]), mae]
answers['Q5'] = mae
assertFloat(answers['Q5'])

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.metrics import precision_score
### Question 6
f = open("beer_5000.json")
dataset = []
for l in f:
    if 'user/gender' in l:
        dataset.append(eval(l))
print(dataset[1])

X = [[1,d['review/text'].count('!')]for d in dataset]
y = [d['user/gender']!='Male' for d in dataset]
model = LogisticRegression()
model.fit(X, y)
y_pred = model.predict(X)
TP, TN, FP, FN = confusion_matrix(y, y_pred).ravel()
a = TP/(TP+FN)
b = TN/(TN+FP)
BER = 1- 1/2*(a+b)
#BER = (FP + FN) / (TP + TN + FP + FN)
answers['Q6'] = [TN, TP, FN, FP, BER]
print(TP, TN, FP, FN, BER)
assertFloatList(answers['Q6'], 5)

### Question 7
model_balanced = LogisticRegression(class_weight='balanced',max_iter=1000)
model_balanced.fit(X, y)

y_pred_b = model_balanced.predict(X)
TP_bal = sum([(a and b) for (a,b) in zip(y_pred_b,y)])
FP_bal = sum([(a and not b) for (a,b) in zip(y_pred_b,y)])
TN_bal = sum([(not a and not b) for (a,b) in zip(y_pred_b,y)])
FN_bal = sum([(not a and b) for (a,b) in zip(y_pred_b,y)])
# TP_bal, TN_bal, FP_bal, FN_bal = confusion_matrix(y, y_pred_balanced).ravel()
c = TP_bal/(TP_bal+FN_bal)
d = TN_bal/(TN_bal+FP_bal)
BER_bal = 1- 1/2*(c+d)

print(TP_bal,TN_bal,FP_bal,FN_bal,BER_bal)
answers['Q7'] = [TP_bal, TN_bal, FP_bal, FN_bal, BER_bal]

assertFloatList(answers['Q7'], 5)

y_probs = model_balanced.predict_proba(X)[:, 1]
ind = list(range(len(y_probs)))
sort_idx = []
while ind:
    mi = ind[0]
    for i in ind:
        if y_probs[i] > y_probs[mi]:
            mi = i
    sort_idx.append(mi)
    ind.remove(mi)

y_t_sort = [y[i] for i in sort_idx]

pk= []

for K in [1, 10, 100, 1000, 10000]:
    pk.append(precision_score(y_t_sort[:min(K, len(y_t_sort))],[1] * min(K, len(y_t_sort))))

print(pk)
answers['Q8'] = pk

f = open("answers_hw1.txt", 'w')
f.write(str(answers) + '\n')
f.close()


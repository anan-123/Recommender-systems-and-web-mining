

import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy
import string
import random
import string
from sklearn import linear_model
import warnings
warnings.filterwarnings("ignore")
def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N
def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)
def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r

answers = {}
# Some data structures that will be useful
allRatings = []
for l in readCSV("train_Interactions.csv.gz"):
    allRatings.append(l)
len(allRatings)
ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))

##################################################
# Read prediction                                #
##################################################
# Copied from baseline code
bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in readCSV("train_Interactions.csv.gz"):
    bookCount[book] += 1
    totalRead += 1

mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > totalRead/2: break

### Question 1
import random
import numpy as np

best_t = 0
best_a = float('-inf')
total_c = sum(index_c for index_c, _ in mostPopular)
neg = []
valid = []
pred =0
for u, b, r in ratingsValid:
    unrated_b = set(bookCount.keys()) - {book for book,c in ratingsPerUser[u]}
    neg.append((u,random.choice(list(unrated_b)), 0))

for u, b, c in ratingsValid:
    valid.append((u, b, 1))
valid.extend(neg)

pred = sum((1 if b in return1 else 0) == a for u, b, a in valid)

answers['Q1'] = pred/len(valid)
print(answers['Q1'])
assertFloat(answers['Q1'])

### Question 2

for thresh in np.arange(0.4, 0.6, 0.01):
    sb = set()
    count = 0
    for index_c, book in mostPopular:
        count += index_c
        sb.add(book)
        if count >= total_c*thresh:
            break
    predictions = [(1 if b in sb else 0) for u,b,a in valid]
    acc = sum(1 for pred, a in zip(predictions, [c for a,b,c in valid])if pred==a)/len(valid)
    if acc > best_a:
      best_t = thresh
      best_a = acc

answers['Q2'] = [best_t, best_a]
print(answers['Q2'])
assertFloat(answers['Q2'][0])
assertFloat(answers['Q2'][1])

best_t = 0
best_a = float('-inf')
total_c = sum(index_c for index_c, _ in mostPopular)
neg = []
valid = []
for u, b, r in ratingsValid:
    unrated_b = set(bookCount.keys()) - {book for book, _ in ratingsPerUser[u]}
    neg.append((u, random.choice(list(unrated_b)), 0))
for u, b, c in ratingsValid:
    valid.append((u, b, 1))
valid.extend(neg)
book_u = defaultdict(set)
for u, b, r in ratingsTrain:
    book_u[b].add(u)
user_sets_per_book = {b: set(u for u, _ in ratingsPerItem[b]) for b in ratingsPerItem}
jaccard_cache = {}

best_t = 0
best_a = float('-inf')

def jaccard(s1, s2):
    return len(s1 & s2) / len(s1 | s2)

for t in np.arange(0.0, 0.5, 0.001):
    correct = 0
    total = 0
    for u, b, act in valid:
        max_jacc = 0
        if ratingsPerUser[u]:
            for b_p,c in ratingsPerUser[u]:
                max_jacc = max(max_jacc, jaccard(book_u[b], book_u[b_p]))
        pred = 1 if max_jacc > t else 0
        if pred == act:
            correct += 1
        total += 1
    acc = correct / total
    if acc > best_a:
        best_t = t
        best_a = acc
        if best_a > 0.705: # if accuracy good enough stop the loop because we use 0.001 steps it takes very long time to finish. =
            break
        print(best_a)

answers['Q3'] = best_a
print(answers['Q3'])
assertFloat(answers['Q3'])

def jaccard2(b1, b2):
    if (b1, b2) in jaccard_cache:
        return jaccard_cache[(b1, b2)]
    u1 = user_sets_per_book.get(b1, set())
    if (b2, b1) in jaccard_cache:
        return jaccard_cache[(b2, b1)]
    u2 = user_sets_per_book.get(b2, set())
    if not u1 or not u2:
        return 0
    else:
        jaccard_cache[(b1, b2)] = jaccard(u1,u2)
        return jaccard(u1,u2)

def eval_combined(thresh, popular_books):
    correct = 0
    for u,b,act in valid:
        max_jacc = 0
        if ratingsPerUser[u]:
            max_jacc = max(jaccard2(b,b_p) for b_p,c in ratingsPerUser[u])
        pred = 1 if max_jacc > thresh or b in popular_books else 0
        if pred == act:
            correct += 1
    return correct / len(valid)

best_combined_acc = eval_combined(0.59, return1)

answers['Q4'] = best_combined_acc
print(answers['Q4'])
assertFloat(answers['Q4'])

with open("predictions_Read.csv", 'w') as pred_file:
    for line in open("pairs_Read.csv"):
        if line.startswith("userID"):
            pred_file.write(line)
            continue
        u, b = line.strip().split(',')
        max_jacc = 0
        if ratingsPerUser[u]:
            max_jacc = max(jaccard2(b, b_prime) for b_prime, _ in ratingsPerUser[u])
        pred = 1 if max_jacc > 0.59 or b in return1 else 0
        pred_file.write(f"{u},{b},{pred}\n")

answers['Q5'] = "I confirm that I have uploaded an assignment submission to gradescope"
assert isinstance(answers['Q5'], str)
print(answers['Q5'])

import numpy as np
train_data = ratingsTrain
valid_data = ratingsValid

u2i = {u: idx for idx, u in enumerate(set(u for u, _, _ in train_data))}
i2u = {i: idx for idx, i in enumerate(set(i for _, i, _ in train_data))}

alpha = np.random.rand() * 0.01
beta_u = np.random.rand(len(u2i)) * 0.01
beta_i = np.random.rand(len(i2u)) * 0.01

lamb = 1
lr = 1e-6
epochs = 100

def compute_mse_and_grads():
    mse = 0
    grad_alpha = 0
    grad_beta_u = np.zeros(len(u2i))
    grad_beta_i = np.zeros(len(i2u))
    for u, i, r in train_data:
        if u in u2i and i in i2u:
            r_hat = alpha + beta_u[u2i[u]] + beta_i[i2u[i]]
            mse += (r - r_hat)**2
            grad_alpha -= 2*(r - r_hat)
            grad_beta_u[u2i[u]]-=2*(r - r_hat)
            grad_beta_i[i2u[i]]-=2*(r - r_hat)
    mse+= lamb*(np.sum(beta_u**2)+np.sum(beta_i**2))
    grad_alpha+=2*lamb*alpha
    grad_beta_u+=2*lamb*beta_u
    grad_beta_i+=2*lamb*beta_i
    return mse, grad_alpha, grad_beta_u, grad_beta_i

def gradient_descent():
    global alpha, beta_u, beta_i
    prev_mse = float('inf')
    for epoch in range(epochs):
        mse, grad_alpha, grad_beta_u, grad_beta_i = compute_mse_and_grads()
        if abs(prev_mse - mse) < 1e-6:
            break
        alpha-=lr*grad_alpha
        prev_mse=mse
        beta_u-=lr*grad_beta_u
        beta_i-=lr*grad_beta_i
    return mse

train_mse = gradient_descent()

def compute_valid_mse():
    mse = 0
    for u, i, r in valid_data:
        if u in u2i and i in i2u:
          r_hat = alpha + beta_u[u2i[u]] + beta_i[i2u[i]]
          mse += (r - r_hat) ** 2
    mse/=len(valid_data)
    return mse

valid_mse = compute_valid_mse()

answers['Q6'] = valid_mse
assertFloat(answers['Q6'])
print(answers['Q6'])

answers['Q7'] = [str(list(u2i.keys())[np.argmax(beta_u)]),str(list(u2i.keys())[ np.argmin(beta_u)]),float(beta_u[np.argmax(beta_u)]),float(beta_u[ np.argmin(beta_u)])]
assert [type(x) for x in answers['Q7']] == [str, str, float, float]
print(answers['Q7'])

best_lambda = None
best_valid_mse = float('inf')
def gradient_descent_batch(train_data,u2i,i2u,alpha,beta_u,beta_i,lr,lamb,epochs=10,batch_size=100):
    for epoch in range(epochs):
        np.random.shuffle(train_data)
        for batch_idx in range(len(train_data)//batch_size):
            batch_data=train_data[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            mse,grad_alpha,grad_beta_u,grad_beta_i =compute_mse_and_grads_batch(batch_data,u2i,i2u,alpha,beta_u,beta_i,lamb)
            alpha-=lr*grad_alpha
            beta_i-=lr*grad_beta_i
            beta_u-=lr*grad_beta_u
    return alpha,beta_u,beta_i

def compute_mse_and_grads_batch(batch_data,u2i,i2u,alpha,beta_u,beta_i,lamb):
    grad_alpha=0
    grad_beta_u=np.zeros(len(u2i))
    mse = 0
    grad_beta_i=np.zeros(len(i2u))
    for u,i,r in batch_data:
        if u in u2i and i in i2u:
            r_hat=alpha +beta_u[u2i[u]]+ beta_i[i2u[i]]
            mse+=(r - r_hat)**2
            grad_alpha-= 2*(r - r_hat)
            grad_beta_u[u2i[u]]-=2*(r - r_hat)
            grad_beta_i[i2u[i]]-=2*(r - r_hat)
    mse+=lamb*(np.sum(beta_u**2) + np.sum(beta_i**2))
    grad_alpha += 2*lamb*alpha
    grad_beta_i+= 2*lamb *beta_i
    grad_beta_u+=2*lamb*beta_u
    mse/=len(batch_data)
    return mse,grad_alpha,grad_beta_u,grad_beta_i

def compute_valid_mse(valid_data, u2i, i2u, alpha, beta_u, beta_i):
    mse = 0
    for u,i,r in valid_data:
        if u in u2i and i in i2u:
          r_hat =alpha +beta_u[u2i[u]]+beta_i[i2u[i]]
          mse+=(r - r_hat) ** 2
    mse/=len(valid_data)
    return mse

for lamb in  [0.01,0.1,1,10,100]:
    alpha= np.random.randn()*0.001
    beta_u=np.random.randn(len(u2i))*0.001
    beta_i= np.random.randn(len(i2u))*0.001
    alpha,beta_u, beta_i = gradient_descent_batch(train_data,u2i,i2u,alpha,beta_u,beta_i,1e-4,lamb,10)
    valid_mse = compute_valid_mse(valid_data, u2i, i2u, alpha, beta_u, beta_i)
    if valid_mse < best_valid_mse:
        best_valid_mse = valid_mse
        best_lambda = lamb

answers['Q8'] = (best_lambda, best_valid_mse)
print(answers['Q8'])
assertFloat(answers['Q8'][0])
assertFloat(answers['Q8'][1])

print(answers)

predictions = open("predictions_Rating.csv", 'w')
for l in open("pairs_Rating.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
predictions.close()

f = open("answers_hw3.txt", 'w')
f.write(str(answers) + '\n')
f.close()


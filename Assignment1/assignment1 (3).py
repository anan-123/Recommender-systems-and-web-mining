

import gzip
from collections import defaultdict

def readGz(path):
  for l in gzip.open(path, 'rt'):
    yield eval(l)

def readCSV(path):
  f = gzip.open(path, 'rt')
  f.readline()
  for l in f:
    yield l.strip().split(',')

### Rating baseline: compute averages for each user, or return the global average if we've never seen the user before

allRat_train = []
allRat_test = []
userRat_train = defaultdict(list)
userRat_test = defaultdict(list)
subset_count = 10000
i = 0
for user,book,r in readCSV("train_Interactions.csv.gz"):
  if i >= subset_count:
    break
  r = int(r)
  if i < subset_count * 0.8: # split to train and validation
    allRat_train.append(r)
    userRat_train[user].append(r)
  else:
    allRat_test.append(r)
    userRat_test[user].append(r)

  i = i+1

# print(len(allRat_train))
# print(len(allRat_test))
# print(userRat_train)

# !pip install scikit-surprise
import gzip
from collections import defaultdict
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import heapq
from surprise import SVD, Reader, Dataset
from surprise.model_selection import GridSearchCV
from surprise import accuracy
import pandas as pd

# Getting subset data from original sets
allRatings = []
import random
for l in readCSV("train_Interactions.csv.gz"):
    allRatings.append(l)
len(allRatings)
subset_count = 200000
train_data_svd = pd.read_csv('train_Interactions.csv.gz')

bookCount_train = defaultdict(int)
totalRead_train = 0
bookCount_valid = defaultdict(int)
totalRead_valid = 0
book_rat = defaultdict(list) # ratings for each book
user_rat = defaultdict(list) # ratings for user
# ratingsTrain = allRatings[:int(subset_count*0.8)]
ratingsTrain = allRatings[:]
ratingsTrain_temp = ratingsTrain
pairs_df = pd.read_csv('pairs_Rating.csv')
ratingsValid = allRatings[int(subset_count*0.8):subset_count]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
user_b = defaultdict(list) # contains all books read by 1 user
book_u = defaultdict(list) # contains all users who read the book

for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    bookCount_train[b]+=1
    totalRead_train+=1
    book_u[b].append(u)
    book_rat[b].append(float(r))
    ratingsPerItem[b].append((u,r))
    user_b[u].append(b)
    user_rat[u].append(float(r))

for u,b,r in ratingsValid:
    bookCount_valid[b]+=1
    totalRead_valid+=1

# adding samples for which the user has not read the book
# import random

# neg = []
# valid = []

# # Set of all books
# allBooks = set(bookCount_train.keys())
# # for train
# neg_train = []
# for u, b, r in ratingsTrain:
#     unrated_b = set(bookCount_valid.keys()) - {book for book,c in ratingsPerUser[u]}
#     neg_train.append((u,random.choice(list(unrated_b)), 0))

# ratingsTrain = ratingsTrain+neg_train
# # for valid
# for u, b, r in ratingsValid:
#     unrated_b = set(bookCount_valid.keys()) - {book for book,c in ratingsPerUser[u]}
#     neg.append((u,random.choice(list(unrated_b)), 0))
# ratingsValid = ratingsValid+neg



"""### Part 1: Predict Read
Models tried:
1. SVM
2. Gradient boost classifier
3. Jaccard similarity - homework3 based solution
"""

# creating the feature set: experimented with mean,min,max,cosine,jaccard, total books, total users etc.
feature = {}
b_pop = {}
b_avg = {}
u_avg = {}
u_sum = defaultdict(float)
u_count = defaultdict(int)

for u, b, r in ratingsTrain_temp:
    r = float(r)
    b_pop[b] = b_pop.get(b, 0) + 1
    b_avg[b] = b_avg.get(b, 0) + r
    u_sum[u] += r
    u_count[u] += 1
for b in b_avg:
    b_avg[b] /= b_pop[b]
u_avg = {u: u_sum[u] / u_count[u] for u in u_sum}

import heapq

heap = [(-p,b) for b,p in b_pop.items()]
heapq.heapify(heap)

popular_books, count, total = set(), 0, sum(b_pop.values())

while count <= total / 2:
    pop, book = heapq.heappop(heap)
    popular_books.add(book)
    count -= pop
# print(count)
# print(popular_books)

import numpy as np
import heapq
from collections import defaultdict

# Initialize dictionaries for book popularity, average ratings, etc.
b_pop = {}
b_avg = {}
u_avg = {}
u_sum = defaultdict(float)
u_count = defaultdict(int)
book_ratings = defaultdict(list)
user_ratings = defaultdict(list)

# Process the training data to compute popularity, averages, etc.
for u, b, r in ratingsTrain_temp:
    r = float(r)
    b_pop[b] = b_pop.get(b, 0) + 1
    b_avg[b] = b_avg.get(b, 0) + r
    u_sum[u] += r
    u_count[u] += 1
    book_ratings[b].append(r)
    user_ratings[u].append(r)

# Calculate average ratings for books and users
for b in b_avg:
    b_avg[b] /= b_pop[b]

u_avg = {u: u_sum[u] / u_count[u] for u in u_sum}

# Create the heap to select popular books
heap = [(-p, b) for b, p in b_pop.items()]
heapq.heapify(heap)

# Collect the popular books
popular_books, count, total = set(), 0, sum(b_pop.values())
while count <= total / 2:
    pop, book = heapq.heappop(heap)
    popular_books.add(book)
    count -= pop

X_read, y_read = [], []
feat = {
    'b_pop': 0,  # Book popularity
    'b_avg': 0,  # Book average rating
    'b_var': 0,  # Book rating variance
    'pop': 0,    # Is book popular
    'u_count': 0, # User-rated books count
    'u_ratio': 0, # User books to total ratio
    'u_avg': 0,   # User average rating
    'u_var': 0    # User rating variance
}



for u, b, r in ratingsTrain:
    u_books = user_b.get(u, set())
    u_books_count = len(u_books)
    u_ratio = u_books_count / len(book_u) if u_books_count > 0 else 0
    u_rating_avg = u_avg.get(u, np.mean(list(u_avg.values())))
    u_rating_var = np.var(user_ratings.get(u, [])) if len(user_ratings.get(u, [])) > 1 else 0

    feat['b_pop'] = b_pop.get(b, 0)
    feat['b_avg'] = b_avg.get(b, 0)
    feat['b_var'] = np.var(book_ratings.get(b, [])) if len(book_ratings.get(b, [])) > 1 else 0
    feat['pop'] = 1 if b in popular_books else 0
    feat['u_count'] = u_books_count
    feat['u_ratio'] = u_ratio
    feat['u_avg'] = u_rating_avg
    feat['u_var'] = u_rating_var
    X_read.append(list(feat.values()))
    y_read.append(1)

    # Negative sample
    neg_b = np.random.choice(list(book_ratings.keys()))
    while neg_b in u_books:
        neg_b = np.random.choice(list(book_ratings.keys()))
    feat['b_pop'] = b_pop.get(neg_b, 0)
    feat['b_avg'] = b_avg.get(neg_b, 0)
    feat['b_var'] = np.var(book_ratings.get(neg_b, [])) if len(book_ratings.get(neg_b, [])) > 1 else 0
    feat['pop'] = 1 if neg_b in popular_books else 0
    feat['u_count'] = u_books_count
    feat['u_ratio'] = u_ratio
    feat['u_avg'] = u_rating_avg
    feat['u_var'] = u_rating_var
    X_read.append(list(feat.values()))
    y_read.append(0)



with open("store_data_iterim.txt",'w') as f:
    f.write(" Train after appending negative samples\n")
    f.write(str(ratingsTrain))
    f.write("\n Features \n")
    f.write(str(X_read))
    f.write("\n labels \n")
    f.write(str(y_read))



with open("features.txt",'w') as f:
    f.write(str(b_pop))
    f.write("\n")
    f.write(str(b_avg))
    f.write("\n")
    f.write(str(u_sum))
    f.write("\n")
    f.write(str(u_count))
    f.write("\n")
    f.write(str(book_ratings))
    f.write("\n")
    f.write(str(b_pop))
    f.write("\n")

# # Reading from store_data_iterim.txt
# with open("store_data_iterim.txt", 'r') as f:
#     lines = f.readlines()

# # Extract and parse variables
# ratingsTrain = eval(lines[1])  # Assuming the second line contains the ratingsTrain data
# X_read = eval(lines[3])        # Assuming the fourth line contains the X_read data
# y_read = eval(lines[5])        # Assuming the sixth line contains the y_read data

# print("RatingsTrain:", ratingsTrain)
# print("Features (X_read):", X_read)
# print("Labels (y_read):", y_read)

len(y_read)

# Train-test split (same as before)
X_train_read, X_val_read, y_train_read, y_val_read = train_test_split(X_read, y_read, test_size=0.2, random_state=42)
print(np.unique(y_train_read))
print(np.unique(y_val_read))

# SVM implementation: better accuracy when tried on subset of 10000 but for 200000 takes very long time doesn't finish.

# # SVM with a pipeline
# model_read = make_pipeline(
#     StandardScaler(),
#     SVC(kernel='linear', C=1, random_state=42)  # You can choose different kernels (linear, poly, rbf)
# )

# model_read.fit(X_train_read, y_train_read)

# val_read_predictions = model_read.predict(X_val_read)

# read_accuracy = accuracy_score(y_val_read, val_read_predictions)
# print(f"Read Prediction Accuracy: {read_accuracy}")
# cv_scores = cross_val_score(model_read, X_read, y_read, cv=5, scoring='accuracy')
# print(f"Cross-validation Accuracy: {cv_scores.mean()}")

X_train, X_val, y_train, y_val = train_test_split(X_read, y_read, test_size=0.2, random_state=42)

model = make_pipeline(
    StandardScaler(),
    GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
)

model.fit(X_train, y_train)
y_pred = model.predict(X_val)

acc = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {acc:.4f}")
cv_acc = cross_val_score(model, X_read, y_read, cv=5, scoring='accuracy').mean()

print(f"Cross-validation Accuracy: {cv_acc:.4f}")

with open("predictions_Read.csv", 'w') as predictions:
    with open("pairs_Read.csv", 'r') as pairs:

        for line in pairs:
            if line.startswith("userID"):
                predictions.write(line)
                continue

            u, b = line.strip().split(',')
            u_books = user_b.get(u, set())
            u_books_count = len(u_books)
            u_ratio = u_books_count / len(book_u) if u_books_count > 0 else 0
            u_rating_avg = u_avg.get(u, np.mean(list(u_avg.values())))
            u_rating_var = np.var(user_ratings.get(u, [])) if len(user_ratings.get(u, [])) > 1 else 0

            feat['b_pop'] = b_pop.get(b, 0)
            feat['b_avg'] = b_avg.get(b, 0)
            feat['b_var'] = np.var(book_ratings.get(b, [])) if len(book_ratings.get(b, [])) > 1 else 0
            feat['pop'] = 1 if b in popular_books else 0
            feat['u_count'] = u_books_count
            feat['u_ratio'] = u_ratio
            feat['u_avg'] = u_rating_avg
            feat['u_var'] = u_rating_var


            features = np.array(list(feat.values())).reshape(1, -1)

            predicted_rating = model.predict(features)[0]

            predictions.write(f"{u},{b},{int(predicted_rating)}\n")

"""### Part2: Predict rating : using SVD and grid search for optimal hyperparams
Models tried:
1. Latent factor model
    * batch and regular gradient descent: 1.76, 1.74
    * with gradient descent : 1.7 and with alternate changes without gradient descent: 1.64
    * after hyperparams tuning for gradient descent: 1.70
2. SVD : hyperparameter tuned form 1.45 accuracy to 1.42
    * tried training with negative samples. RMSE reduced to 0.7 but the accuracy on leaderboard is too high
3. SVM using one hot and ordinal encoding
4. Homework3 baseline
5. Derived features with svd in Random forest model
"""

data = Dataset.load_from_df(train_data_svd[['userID', 'bookID', 'rating']],reader = Reader(rating_scale=(1, 5)))

param_grid = {
    'n_factors': [50,52,48],
    'n_epochs': [40,42,38],
    'lr_all': [0.002, 0.005, 0.01],
    'reg_all': [0.4, 0.6, 0.8]
}


grid_search = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
grid_search.fit(data)
best_params = grid_search.best_params['rmse']
svd_model = SVD(**best_params)
svd_model.fit(data.build_full_trainset())
for _, row in pairs_df.iterrows():
  #  print(u,b)
    pred = svd_model.predict(row['userID'], row['bookID'])
    prediction_list.append([u, b, pred.est])

pred = pd.DataFrame(prediction_list, columns=["userID", "bookID", "prediction"])
pred.to_csv('predictions_Rating.csv', index=False)


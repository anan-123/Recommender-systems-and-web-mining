# Recommender-systems-and-web-mining
Assignments for UCSD CSE 258 Web Mining and Recommender Systems
| **Assignment** | **Description** |
|----------------|-----------------|
| **Assignment 2** | Predicting Users’ Makeup Rating on Ulta Products: Did data analysis, compared the performance on 7 models and gave the best model for rating prediction |
| **Assignment 1** | Built recommender systems for book reviews from Goodreads, focusing on two tasks: predicting whether a user would read a book and predicting star ratings for books. A Gradient Boosting Classifier achieved the best performance for predicting whether a user would read a book, with an accuracy of 0.7583. For rating prediction, hyperparameter-tuned SVD yielded the best accuracy, improving from RMSE = 1.45 to 1.41.|
| **Homework 3** | Improved the read and rating prediction tasks from Assignment 1 using techniques like Jaccard similarity and bias-based rating prediction. |
| **Homework 2** | Worked on model pipelines and diagnostics using bankruptcy data and implemented recommendation systems using Goodreads book review data. |
| **Homework 1** | Explored regression and classification tasks using book review data from Goodreads and beer review data, focusing on predicting ratings and user gender. |

## **Assignment 1**

### **Flow of Code:**

#### **Part 1: Predicting Read Prediction**
1. **Data Preparation:**
   - Generated training, testing, and user-per-book lists, as well as other necessary lists.
   - Defined features such as popularity and mean rating in a feature dictionary (`feat dict`).
   - Populated `X` (features) and `y` (labels) for the model using user, book, and ratings in the rating training data. Negative samples were also added, doubling the length of the training data.
2. **Model Training:**
   - Split the data into train and test sets.
   - Trained a **Gradient Boosting Classifier** model. Gradient Boosting builds decision trees sequentially to correct errors made by previous trees, reducing bias and improving predictions.
3. **Prediction:**
   - For each pair in `pair_Read.csv`, the predictions were written to `predictions_Read.csv`.

#### **Part 2: Predicting Rating**
1. **Data Conversion:**
   - The `train.csv` and `pair_rating.csv` were read in the first step of Part 1.
   - Converted the training data into the Surprise SVD format.
2. **Hyperparameter Tuning:**
   - Performed a **Grid Search** to find optimal hyperparameters, experimenting with:
     - `n_factors`: [50, 52, 48, 100, 200, 25, 30]
     - `n_epochs`: [40, 42, 38, 20, 30, 50, 100]
     - `lr_all`: [0.002, 0.005, 0.01, 0.1, 1]
     - `reg_all`: [0.4, 0.6, 0.8]
3. **Model Training:**
   - Trained the **SVD** model using the best parameters obtained from the grid search.
4. **Prediction:**
   - Wrote the predictions using the `model.predict` method to `predictions_Rating.csv`.

### **Different Approaches Used:**
#### **Part 2: Predicting Rating**
1. **Latent Factor Model Implementation from Scratch:**
   - Implemented latent factor models and tested different gradient descent approaches, including:
     - Batch and regular gradient descent: RMSE = 1.76 and 1.74.
     - Two approaches with gradient descent and alternate changes without it: RMSE = 1.70 and 1.64.
     - Hyperparameter tuning for gradient descent: RMSE = 1.70.
2. **SVD Model:**
   - Hyperparameter tuning for SVD improved the accuracy from RMSE = 1.45 to RMSE = 1.41 (best accuracy achieved).
   - Training with negative samples resulted in RMSE = 0.7, but leaderboard accuracy was low, suggesting overfitting.
3. **SVM:**
   - Used both **one-hot** and **ordinal encoding** for SVM-based predictions.
4. **Random Forest with SVD Features:**
   - Achieved excellent RMSE but poor leaderboard accuracy, indicating potential overfitting.
5. **Baseline Approach (Homework 3):**
   - Implemented a baseline approach from Homework 3.

#### **Part 1: Predicting Read Prediction**
1. **Models Tried:**
   - **SVM** for classification.
   - **Gradient Boost Classifier** (best performing model):
     - Validation Accuracy: 0.7583
     - Cross-validation Accuracy: 0.7584
   - **Jaccard Similarity** (based on Homework 3 solution): Accuracy = 0.71.

### **Summary of Results:**
- The **Gradient Boosting Classifier** yielded the best accuracy for predicting read status (0.7583), while **SVD with hyperparameter tuning** was the top model for predicting ratings, achieving an RMSE of 1.41.
- **Overfitting** was observed with models like **SVD with negative samples** and **Random Forest with derived features**, where RMSE was good, but leaderboard accuracy was low.

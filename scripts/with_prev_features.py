import json
import numpy as np
from sklearn.feature_selection import mutual_info_classif, GenericUnivariateSelect
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
import datetime
from statistics import mean, stdev
import imblearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

with open("features_652.json", "r") as f:
    data_dict = json.load(f)

data_with_labels = data_dict["0"]
data = []
labels = []

#populating data matrix
for i in range(len(data_with_labels)):
    data.append(data_with_labels[i][0])
    labels.append(data_with_labels[i][1])

data = np.asarray(data)
labels = np.asarray(labels)
fix_data = data
fix_labels = labels

ours = pd.DataFrame(data)
X_df = pd.DataFrame(data)
y_df = pd.DataFrame(labels)
X = X_df.values
Y = y_df.values

# As this is a multiclass classification problem onehotencoding our Y.
encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()

# splitting data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1,random_state=1, shuffle=True)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

# Step 1: Convert one-hot encoded labels back to numerical labels
y_train = y_train.argmax(axis=1)
y_test = y_test.argmax(axis=1)

x_train_df = pd.DataFrame(x_train)
x_test_df = pd.DataFrame(x_test)

y_train_df = pd.DataFrame(y_train)
y_test_df = pd.DataFrame(y_test)

# Step 2: Create an SVM classifier
svm_classifier = svm.SVC(kernel='rbf', C=100)# You can choose different parameters based on your data

# Step 3: Train the SVM classifier
svm_classifier.fit(x_train, y_train)

# Step 4: Make predictions on the test set
y_test_pred = svm_classifier.predict(x_test)

# Step 5: Evaluate the model on the test set
accuracy = accuracy_score(y_test, y_test_pred)
print("Test Accuracy:", accuracy)

from mealpy import IntegerVar, GA
import numpy as np

import xgboost as xgb
from sklearn.metrics import accuracy_score

def objective_xgb(features):
    # print(features)
    features = np.unique(features)
    X_train_subset = x_train_df.iloc[:, features]
    X_test_subset = x_test_df.iloc[:, features]

    xgboost_classifier = xgb.XGBClassifier(n_estimators=5, random_state=42)
    xgboost_classifier.fit(X_train_subset, y_train)

    y_pred = xgboost_classifier.predict(X_test_subset)
    acc = accuracy_score(y_test, y_pred)

    return acc

def objective_svm(features):
    X_train_subset = x_train_df.iloc[:, features]
    X_test_subset = x_test_df.iloc[:, features]
    svm_classifier_rbf = svm.SVC(kernel='rbf', C=100)
    svm_classifier_rbf.fit(X_train_subset, y_train)
    y_pred = svm_classifier_rbf.predict(X_test_subset)
    acc = accuracy_score(y_test, y_pred)
    return acc

svclassifier = svm.SVC(kernel='rbf', C=100)
knn_classifier = KNeighborsClassifier(n_neighbors=1)

def objective_svm_strat(features):
    features = features.astype(int)
    kkkkk = fix_data[:,features]

    oversample = imblearn.over_sampling.SMOTE()
    imp, labels = oversample.fit_resample(kkkkk, fix_labels)

    sc = StandardScaler()
    x_scaled = sc.fit_transform(imp)

    y = np.asarray(labels)

    # Create StratifiedKFold object.
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    lst_accu_stratified = []

    for train_index, test_index in skf.split(x_scaled, labels):
        x_train_fold, x_test_fold = x_scaled[train_index], x_scaled[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        svclassifier.fit(x_train_fold, y_train_fold)
        lst_accu_stratified.append(svclassifier.score(x_test_fold, y_test_fold))

    return mean(lst_accu_stratified)


def objective_knn_strat(features):
    features = features.astype(int)
    kkkkk = fix_data[:,features]

    oversample = imblearn.over_sampling.SMOTE()
    imp, labels = oversample.fit_resample(kkkkk, fix_labels)

    sc = StandardScaler()
    x_scaled = sc.fit_transform(imp)

    y = np.asarray(labels)

    # Create StratifiedKFold object.
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    lst_accu_stratified = []

    for train_index, test_index in skf.split(x_scaled, labels):
        x_train_fold, x_test_fold = x_scaled[train_index], x_scaled[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        knn_classifier.fit(x_train_fold, y_train_fold)
        lst_accu_stratified.append(knn_classifier.score(x_test_fold, y_test_fold))

    return mean(lst_accu_stratified)

problem_dict = {
    "obj_func": objective_knn_strat,
    "bounds": IntegerVar(lb=[0, ] * 20, ub=[651, ] * 20,),
    "minmax": "max",
}

optimizer = GA.BaseGA(epoch=100, pop_size=256, pc=0.95, pm=0.1)
optimizer.solve(problem_dict)

print(optimizer.g_best.solution)
print(optimizer.g_best.target.fitness)

main_sol = optimizer.g_best.solution
main_sol_int = main_sol.astype(int)

print(len(main_sol_int))
np.save("beat_20.npy",main_sol_int)
print("Solution\n")
print(main_sol_int)


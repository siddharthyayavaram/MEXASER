import pandas as pd
import numpy as np

import os
import sys

import matplotlib.pyplot as plt

from statistics import mean, stdev
import imblearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Features.head()
# Features.to_csv('savee_features.csv', index=False)

Features = pd.read_csv('tess_features.csv')
Features.head()

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

X = Features.iloc[: ,:-1].values
Y = Features['labels'].values
encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()

labels=Y
oversample = imblearn.over_sampling.SMOTE()
data, labels = oversample.fit_resample(X, Y)

x_train, x_test, y_train, y_test = train_test_split(data, labels,test_size=0.1,random_state=4, shuffle=True)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

y_train = y_train.argmax(axis=1)
y_test = y_test.argmax(axis=1)
x_train_df = pd.DataFrame(x_train)
x_test_df = pd.DataFrame(x_test)
y_train_df = pd.DataFrame(y_train)
y_test_df = pd.DataFrame(y_test)


from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score

svm_classifier = svm.SVC(kernel='rbf', C=100) 
svm_classifier.fit(x_train, y_train)

# Step 4: Make predictions on the test set
y_test_pred = svm_classifier.predict(x_test)

# Step 5: Evaluate the model on the test set
precision = accuracy_score(y_test, y_test_pred)
print("Test accuracy 1:", precision)

# emotion_dict = {
#     0: 'neutral',
#     1: 'calm',
#     2: 'happy',
#     3: 'sad',
#     4: 'angry',
#     5: 'fearful',
#     6: 'disgust',
#     7: 'surprised'
# }

# from sklearn.metrics import classification_report
# y_true_emotions = np.array([emotion_dict[label] for label in y_test])
# y_pred_emotions = np.array([emotion_dict[label] for label in y_test_pred])

# # Generate a new classification report
# report_with_emotions = classification_report(y_true_emotions, y_pred_emotions)

# # Display the new classification report
# print(report_with_emotions)

def objective_knn(features):
    features = features.astype(int)
    X_train_subset = x_train_df.iloc[:, features]
    X_test_subset = x_test_df.iloc[:, features]
    X_test_subset = np.array(X_test_subset)  # Ensure X_test_subset is a NumPy array
    if not X_test_subset.flags.c_contiguous:  # Check contiguity and make it contiguous if necessary
        X_test_subset = np.ascontiguousarray(X_test_subset)
    knn_classifier = KNeighborsClassifier(n_neighbors=1)
    knn_classifier.fit(X_train_subset, y_train)
    y_pred = knn_classifier.predict(X_test_subset)
    acc = accuracy_score(y_test, y_pred)
    return acc

# objective_knn(np.array(list(range(6372))))

from sklearn.feature_selection import mutual_info_classif

# objective_info_gain(np.array(list(range(100))))

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
import datetime
from statistics import mean, stdev
import imblearn

from mealpy import IntegerVar,GA,DE,CRO,EP,HS,WCA,AEO,GCO,FA,QSA,SHADE

def objective_knn(features):
    features = features.astype(int)
    X_train_subset = x_train_df.iloc[:, features]
    X_test_subset = x_test_df.iloc[:, features]
    X_test_subset = np.array(X_test_subset)  # Ensure X_test_subset is a NumPy array
    if not X_test_subset.flags.c_contiguous:  # Check contiguity and make it contiguous if necessary
        X_test_subset = np.ascontiguousarray(X_test_subset)
    knn_classifier = KNeighborsClassifier(n_neighbors=1)
    knn_classifier.fit(X_train_subset, y_train)
    y_pred = knn_classifier.predict(X_test_subset)
    acc = accuracy_score(y_test, y_pred)
    return acc

sv_classifier = svm.SVC(kernel='rbf', C=100)
knn_classifier = KNeighborsClassifier(n_neighbors=1)

def objective_1(features):
    Y = Features['labels'].values
    features = features.astype(int)

    data_1 = X[:,features]

    labels=Y

    oversample = imblearn.over_sampling.SMOTE()
    data, labels = oversample.fit_resample(data_1, Y)

    sc = StandardScaler()
    x_scaled = sc.fit_transform(data)

    y = np.asarray(labels)

    # Create StratifiedKFold object.
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=4)
    lst_accu_stratified = []
    # y_actual_list = []
    # y_pred_list = []

    for train_index, test_index in skf.split(x_scaled, labels):
        x_train_fold, x_test_fold = x_scaled[train_index], x_scaled[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        knn_classifier.fit(x_train_fold, y_train_fold)
        lst_accu_stratified.append(knn_classifier.score(x_test_fold, y_test_fold))

    return mean(lst_accu_stratified)

def objective_2(features):
    Y = Features['labels'].values
    features = features.astype(int)

    data_1 = X[:,features]

    labels=Y

    oversample = imblearn.over_sampling.SMOTE()
    data, labels = oversample.fit_resample(data_1, Y)

    sc = StandardScaler()
    x_scaled = sc.fit_transform(data)

    y = np.asarray(labels)

    # Create StratifiedKFold object.
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=4)
    lst_accu_stratified = []
    # y_actual_list = []
    # y_pred_list = []

    for train_index, test_index in skf.split(x_scaled, labels):
        x_train_fold, x_test_fold = x_scaled[train_index], x_scaled[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        sv_classifier.fit(x_train_fold, y_train_fold)
        lst_accu_stratified.append(sv_classifier.score(x_test_fold, y_test_fold))

    return mean(lst_accu_stratified)

def objective_info_gain(features):
    features = features.astype(int)
    X_train_subset = x_train_df.iloc[:, features]
    info_gain = mutual_info_classif(X_train_subset, y_train)  # Calculate information gain for each feature
    return np.sum(info_gain)  # Sum of information gains of selected features

def objective_svm(features):
    features = features.astype(int)
    X_train_subset = x_train_df.iloc[:, features]
    X_test_subset = x_test_df.iloc[:, features]
    svm_classifier_rbf = svm.SVC(kernel='rbf', C=100)
    svm_classifier_rbf.fit(X_train_subset, y_train)
    y_pred = svm_classifier_rbf.predict(X_test_subset)
    acc = accuracy_score(y_test, y_pred)
    return acc

problem_dict = {
    "obj_func": objective_1,
    "bounds": IntegerVar(lb=[0, ] * 50, ub=[6372, ] * 50,),
    "minmax": "max",
}

# optimizer = GA.BaseGA(epoch=100, pop_size=64, pc=0.95, pm=0.1)
# optimizer = CRO.OCRO(epoch=100, pop_size=64, po = 0.4, Fb = 0.9, Fa = 0.1, Fd = 0.1, Pd = 0.5, GCR = 0.1, gamma_min = 0.02, gamma_max = 0.2, n_trials = 5, restart_count = 50)
# optimizer = DE.JADE(epoch=1000, pop_size=50, miu_f = 0.5, miu_cr = 0.5, pt = 0.1, ap = 0.1)  # decent
# optimizer = HS.DevHS(epoch=100, pop_size=64, c_r = 0.95, pa_r = 0.05)

# optimizer =  WCA.OriginalWCA(epoch=150, pop_size=64, nsr = 4, wc = 2.0, dmax = 1e-6) # good WCA
# optimizer = FA.OriginalFA(epoch=100, pop_size=64, max_sparks = 50, p_a = 0.04, p_b = 0.8, max_ea = 40, m_sparks = 50) #   decent
# optimizer =   QSA.ImprovedQSA(epoch=1000, pop_size=64)  alright
# optimizer = SHADE.L_SHADE(epoch=1000, pop_size=50, miu_f = 0.5, miu_cr = 0.5)

# optimizer = AEO.AugmentedAEO(epoch=150, pop_size=64)

optimizer = GA.BaseGA(epoch=50, pop_size=64, pc=0.95, pm=0.1)
optimizer.solve(problem_dict)

print(optimizer.g_best.solution)
print(optimizer.g_best.target.fitness)

import ast
import json
import matplotlib.pyplot as plt

data_str = str(optimizer.history.list_epoch_time[0])

data_str = str(optimizer.history.list_global_best[0])

optimizer.history.list_exploitation

parts = data_str.split(', ')
# Filter for the part containing "Objectives"
objectives_part = [part for part in parts if "Objectives" in part][0]
# Extract the objective value
objective_value = float(objectives_part.split('[')[1].split(']')[0])
print("Objective value:", objective_value)

hist = optimizer.history.list_global_best
cur = optimizer.history.list_current_best
tim = optimizer.history.list_epoch_time
wor = optimizer.history.list_current_worst

# Function to extract objective values from a given list of strings
def extract_objective_values(data_list):
    objectives_values = []
    for i in range(len(data_list)):
        data_str = str(data_list[i])
        parts = data_str.split(',')  # Split the string into parts
        # Find the part containing the objectives
        objectives_part = [part for part in parts if "Objectives" in part][0]
        # Extract the objective value
        objective_value = float(objectives_part.split('[')[1].split(']')[0])
        objectives_values.append(objective_value)
    return objectives_values

# Extract objective values for global best and current best
global_best_objectives = extract_objective_values(hist)
current_best_objectives = extract_objective_values(cur)
current_worst_objectives = extract_objective_values(wor)

epochs = range(1, len(hist) + 1)
plt.figure(dpi=600)  # Set DPI to 300 (adjust as needed)
plt.plot(epochs, global_best_objectives, marker='', label='Global Best')
plt.plot(epochs, current_best_objectives, marker='', label='Current Best')
plt.plot(epochs, current_worst_objectives, marker='', label='Current Worst')
plt.xlabel('Epochs')
plt.ylabel('Objective Value')
plt.title('Objective Value vs Epochs')
plt.legend()
plt.grid(True)

# Save the plot as an image file
plt.savefig('tess_50_knn_1_GA.png')

# Display the plot
plt.show()

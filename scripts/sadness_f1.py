import pandas as pd
import numpy as np
import sys

from statistics import mean, stdev
import imblearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

emotion_dict = {
    4 : 'neutral',
    6 : 'surprised',
    3 : 'happiness',
    5 : 'sadness',
    0 : 'anger',
    2 : 'fear',
    1 : 'disgust',
}

Features = pd.read_csv('savee_features.csv')
Features.head()

X = Features.iloc[: ,:-1].values
Y = Features['labels'].values# As this is a multiclass classification problem onehotencoding our Y.
encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()

# Y = Features['labels'].values
labels=Y
oversample = imblearn.over_sampling.SMOTE()
data, labels = oversample.fit_resample(X, Y)

# sc = StandardScaler()
# x_scaled = sc.fit_transform(data)

# splitting data
x_train, x_test, y_train, y_test = train_test_split(data, labels,test_size=0.1,random_state=42, shuffle=True)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

# scaling our data with sklearn's Standard scaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Step 1: Convert one-hot encoded labels back to numerical labels
y_train = y_train.argmax(axis=1)
y_test = y_test.argmax(axis=1)
x_train_df = pd.DataFrame(x_train)
x_test_df = pd.DataFrame(x_test)
y_train_df = pd.DataFrame(y_train)
y_test_df = pd.DataFrame(y_test)

from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score
# Step 2: Create an SVM classifier
svm_classifier = svm.SVC(kernel='rbf', C=100)  # You can choose different parameters based on your data

# Step 3: Train the SVM classifier
svm_classifier.fit(x_train, y_train)

# Step 4: Make predictions on the test set
y_test_pred = svm_classifier.predict(x_test)

# Step 5: Evaluate the model on the test set
acc = accuracy_score(y_test, y_test_pred)
print("acc:", acc)

from sklearn.metrics import classification_report
y_true_emotions = np.array([emotion_dict[label] for label in y_test])
y_pred_emotions = np.array([emotion_dict[label] for label in y_test_pred])

# Generate a new classification report
report_with_emotions = classification_report(y_true_emotions, y_pred_emotions)

# Display the new classification report
print(report_with_emotions)


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
    f1 = f1_score(y_test, y_pred, average='weighted')
    return f1


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def objective_rf(features):
    features = features.astype(int)
    X_train_subset = x_train_df.iloc[:, features]
    X_test_subset = x_test_df.iloc[:, features]
    random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    random_forest_classifier.fit(X_train_subset, y_train)
    y_pred = random_forest_classifier.predict(X_test_subset)
    acc = accuracy_score(y_test, y_pred)
    return acc

svclassifier = svm.SVC(kernel='rbf', C=100)

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

def objective_1_class(features, class_label=6):
    Y = Features['labels'].values
    features = features.astype(int)

    data_1 = X[:,features]

    labels=Y

    oversample = imblearn.over_sampling.SMOTE()
    data, labels = oversample.fit_resample(data_1, Y)

    sc = StandardScaler()
    x_scaled = sc.fit_transform(data)

    y = np.asarray(labels)

    # knn_classifier = KNeighborsClassifier(n_neighbors=1)  # Initialize KNN classifier here

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=4)
    lst_f1_stratified = []

    for train_index, test_index in skf.split(x_scaled, labels):
        x_train_fold, x_test_fold = x_scaled[train_index], x_scaled[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        knn_classifier.fit(x_train_fold, y_train_fold)
        y_pred = knn_classifier.predict(x_test_fold)
        f1_class = f1_score(y_test_fold, y_pred, average=None)[class_label]
        lst_f1_stratified.append(f1_class)

    return np.mean(lst_f1_stratified)

def objective_svm(features):
    features = features.astype(int)
    X_train_subset = x_train_df.iloc[:, features]
    X_test_subset = x_test_df.iloc[:, features]
    svm_classifier_rbf = svm.SVC(kernel='rbf', C=100)
    svm_classifier_rbf.fit(X_train_subset, y_train)
    y_pred = svm_classifier_rbf.predict(X_test_subset)
    acc = accuracy_score(y_test, y_pred)
    return acc

def objective_knn_class(features, class_label=6):
    # features = features.astype(int)
    X_train_subset = x_train_df.iloc[:, features]
    X_test_subset = x_test_df.iloc[:, features]
    X_test_subset = np.array(X_test_subset)  # Ensure X_test_subset is a NumPy array
    if not X_test_subset.flags.c_contiguous:  # Check contiguity and make it contiguous if necessary
        X_test_subset = np.ascontiguousarray(X_test_subset)
    knn_classifier = KNeighborsClassifier(n_neighbors=1)
    knn_classifier.fit(X_train_subset, y_train)
    y_pred = knn_classifier.predict(X_test_subset)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    f1_class = f1_score(y_test, y_pred, average=None)[class_label]  # Calculate F1 score for the specified class
    return f1_class

problem_dict = {
    "obj_func": objective_1_class,
    "bounds": IntegerVar(lb=[0, ] * 100, ub=[6372, ] *  100,),
    "minmax": "max",
}

optimizer = GA.BaseGA(epoch=150, pop_size=128, pc=0.95, pm=0.1)
optimizer.solve(problem_dict)

print(optimizer.g_best.solution)
print(optimizer.g_best.target.fitness)

feats = optimizer.g_best.solution
# Import library

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Classification Library
from sklearn.tree import DecisionTreeClassifier

# Metrics
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

# Read credit card data file
#df = pd.read_csv("processed.cleveland.data", sep=",")
df = pd.read_csv("creditcard.csv", sep=",")

# class distribution and plotting

print("No Fraud '0': {} % of the dataset".format(round(df['Class'].value_counts(1)[0] * 100,2)))
print("Fraud '1': {} '% of the dataset \n".format(round(df['Class'].value_counts(1)[1] * 100,2)))

colors = ["#0101DF", "#DF0101"]

sns.countplot('Class', data=df, palette=colors)
plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)

# Stratified Sampling
print("Stratified Sampling")
y = df.pop('Class')
X = df
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42, stratify=y)
clf_tree=DecisionTreeClassifier()
clf=clf_tree.fit(X_train,y_train)
print("classification_report_tree",classification_report(y_test,clf_tree.predict(X_test)))

score=accuracy_score(y_test,clf_tree.predict(X_test))
print(score)

# SMOTE Sampling
print("Smote Sampling")

from imblearn.over_sampling import SMOTE
sm = SMOTE()
X_train, y_train = sm.fit_sample(X_train, y_train)

clf_tree=DecisionTreeClassifier()
clf=clf_tree.fit(X_train,y_train)
print("classification_report_tree",classification_report(y_test,clf_tree.predict(X_test)))

score=accuracy_score(y_test,clf_tree.predict(X_test))
print(score)

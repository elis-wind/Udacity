#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.
    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

clf = SVC(gamma='auto', kernel='rbf', C=10000)

### TO SPEED UP TIME and optimize SVC parameters
#features_train = features_train[:int(len(features_train)/100)]
#labels_train = labels_train[:int(len(labels_train)/100)]

clf.fit(features_train, labels_train)
predictions = clf.predict(features_test)

print(accuracy_score(labels_test, predictions))

###Printing the predicted label for 10th, 26th and 50th example
#print(predictions[10])
#print(predictions[26])
#print(predictions[50])

###Printing the numer of predictions for Chris and Sarah
#print("Chris:", np.sum(predictions))
#print("Sarah:", len(predictions) - np.sum(predictions))

#########################################################

#########################################################
'''
You'll be Provided similar code in the Quiz
But the Code provided in Quiz has an Indexing issue
The Code Below solves that issue, So use this one
'''

# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

#########################################################
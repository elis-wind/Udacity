def classify(features_train, labels_train, features_test, labels_test):   
    ### import the sklearn module for GaussianNB
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier
    
    
    ### your code goes here
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score

    clf = GaussianNB()

    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    accuracy = accuracy_score(pred, labels_test)
    print(accuracy)
    #0.884

    return clf
    
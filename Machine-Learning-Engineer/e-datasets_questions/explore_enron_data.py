#!/usr/bin/python3

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).
    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }
    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:
    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import joblib

enron_data = joblib.load(open("../final_project/final_project_dataset.pkl", "rb"))

print("Number of people in the data set: ", len(enron_data))
print("Number of features available for each person: ", len(list(enron_data.values())[0])) 
print("Number of persons of interest in the data set: ", 
    sum(enron_data[pers]['poi'] for pers in enron_data if enron_data[pers]['poi'] == 1))
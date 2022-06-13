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

with open("../final_project/poi_names.txt", 'r') as f:
    poi_names = f.readlines()
    poi_names = poi_names[2:]
    poi_names = [ name.split(' ',1)[1].strip('\n') for name in poi_names]
print(poi_names)

#print(enron_data.keys())
#print(list(enron_data.values())[0])
print("Number of people in the data set: ", len(enron_data))
print("Number of features available for each person: ", len(list(enron_data.values())[0])) 
print("Number of persons of interest in the data set: ", 
    sum(enron_data[pers]['poi'] for pers in enron_data if enron_data[pers]['poi'] == 1))

print("\nThe total value of the stock belonging to James Prentice is ", 
    enron_data["PRENTICE JAMES"]["total_stock_value"])
print("Number of messages from Wesley Colwell to persons of interest: ", 
    enron_data["COLWELL WESLEY"]["from_this_person_to_poi"])
print("The value of stock options exercised by Jeffrey K Skilling is ", 
    enron_data["SKILLING JEFFREY K"]["exercised_stock_options"])

print("\nThe CEO of Enron during much of the time fraud was ongoing: Jeffrey Skilling")
print("The chairman of Enron board of directors during much of the time fraud was ongoing: Kenneth Lay")
print("The CFO of Enron during much of the time fraud was ongoing: Andrew Fastow ")

most_money = max( [ (enron_data[pers]['total_payments'],pers) for pers in ["SKILLING JEFFREY K","LAY KENNETH L", "FASTOW ANDREW S"] ] )
print("Of these three individuals, {} took home the most money ({})\n".format(most_money[1], most_money[0]))

print("Persons with identified salary: ", len([enron_data[pers]['salary'] for pers in enron_data if enron_data[pers]['salary'] != 'NaN']))
print("Persons with identified emails: ", len([enron_data[pers]['email_address'] for pers in enron_data if enron_data[pers]['email_address'] != 'NaN']))
print("POIs with non identified total payments: ", len(poi_names) - len([enron_data[person]['total_payments'] for person in enron_data if (enron_data[person]['poi'])]))
print("All the persons with non identified total payments: ", len([enron_data[person]['total_payments'] for person in enron_data if (enron_data[person]['total_payments']=='NaN')]))
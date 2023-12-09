"""
Read file into texts and calls.
It's ok if you don't understand how to read files.
"""
import csv
with open('texts.csv', 'r') as f:
    reader = csv.reader(f)
    texts = list(reader)

with open('calls.csv', 'r') as f:
    reader = csv.reader(f)
    calls = list(reader)


"""
TASK 1:
How many different telephone numbers are there in the records? 
Print a message:
"There are <count> different telephone numbers in the records."
"""
def extract_numbers(data):
    numbers = []
    for entry in data:
        numbers.extend(entry[:2])

    return numbers


text_numbers = extract_numbers(texts)
call_numbers = extract_numbers(calls)


total_num = text_numbers + call_numbers
count = len(set(total_num))

print(f"There are {count} different telephone numbers in the records.")
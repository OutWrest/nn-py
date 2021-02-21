import csv

with open('data/test.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)
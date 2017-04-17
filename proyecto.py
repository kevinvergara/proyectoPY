# -*- coding: utf-8 -*-

import csv

with open('pima-indians-diabetes.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    #print(readCSV)
    
    for row in readCSV:
        print(row)
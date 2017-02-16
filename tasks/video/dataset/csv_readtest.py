import csv

count = 0
keys = [
    'VideoID',
    'Start',
    'End',
    'WorkerID',
    'Source',
    'AnnotationTime',
    'Language',
    'Description'
]

with open('MSR.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # print([row[v] for v in keys])
        print(row)
        break

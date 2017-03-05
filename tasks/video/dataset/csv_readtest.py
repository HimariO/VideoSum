import csv
import spacy

nlp = spacy.load('en')

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

word_freq = {}

with open('MSR_en.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # print([row[v] for v in keys])
        for i in nlp(row['Description']):
            word = str(i)
            if word in word_freq.keys():
                word_freq[word] += 1
            else:
                word_freq[word] = 0

word_list = [(k, word_freq[k]) for k in word_freq.keys()]

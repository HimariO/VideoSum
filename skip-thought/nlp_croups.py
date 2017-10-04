import spacy

nlp = spacy.load('en')
document = ''

with open('croups.txt') as txt:
    while True:
        line = txt.readline()
        if line == '':
            break
        document += line

print(len(document))

dictionary = {}
filter_dictionary = {}

for word in nlp(document):
    word = str(word)
    try:
        dictionary[word] += 1
    except:
        dictionary[word] = 1

words = list(dictionary.keys())
print(len(words))

for word in words:
    if dictionary[word] >= 3:
        filter_dictionary[word] = dictionary[word]

words = list(filter_dictionary.keys())
print(len(words))

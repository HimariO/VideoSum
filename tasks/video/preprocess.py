import sys
import pickle
import getopt
import numpy as np
from shutil import rmtree
from os import listdir, mkdir
from os.path import join, isfile, isdir, dirname, basename, normpath, abspath, exists
import csv
import random
import spacy

def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

def create_dictionary(files_list):
    """
    creates a dictionary of unique lexicons in the dataset and their mapping to numbers

    Parameters:
    ----------
    files_list: list
        the list of files to scan through

    Returns: dict
        the constructed dictionary of lexicons
    """

    lexicons_dict = {}
    id_counter = 0

    llprint("Creating Dictionary ... 0/%d" % (len(files_list)))

    for indx, filename in enumerate(files_list):
        with open(filename, 'r') as fobj:
            for line in fobj:

                # first seperate . and ? away from words into seperate lexicons
                line = line.replace('.', ' .')
                line = line.replace('?', ' ?')
                line = line.replace(',', ' ')

                for word in line.split():
                    if not word.lower() in lexicons_dict and word.isalpha():
                        lexicons_dict[word.lower()] = id_counter
                        id_counter += 1

        llprint("\rCreating Dictionary ... %d/%d" % ((indx + 1), len(files_list)))

    print("\rCreating Dictionary ... Done!")
    return lexicons_dict


def encode_data(files_list, lexicons_dictionary, length_limit=None):
    """
    encodes the dataset into its numeric form given a constructed dictionary

    Parameters:
    ----------
    files_list: list
        the list of files to scan through
    lexicons_dictionary: dict
        the mappings of unique lexicons

    Returns: tuple (dict, int)
        the data in its numeric form, maximum story length
    """

    files = {}
    story_inputs = None
    story_outputs = None
    stories_lengths = []
    answers_flag = False  # a flag to specify when to put data into outputs list
    limit = length_limit if not length_limit is None else float("inf")

    llprint("Encoding Data ... 0/%d" % (len(files_list)))

    for indx, filename in enumerate(files_list):

        files[filename] = []

        with open(filename, 'r') as fobj:
            for line in fobj:

                # first seperate . and ? away from words into seperate lexicons
                line = line.replace('.', ' .')
                line = line.replace('?', ' ?')
                line = line.replace(',', ' ')

                answers_flag = False  # reset as answers end by end of line

                for i, word in enumerate(line.split()):

                    if word == '1' and i == 0:
                        # beginning of a new story
                        if not story_inputs is None:
                            stories_lengths.append(len(story_inputs))
                            if len(story_inputs) <= limit:
                                files[filename].append({
                                    'inputs':story_inputs,
                                    'outputs': story_outputs
                                })
                        story_inputs = []
                        story_outputs = []

                    if word.isalpha() or word == '?' or word == '.':
                        if not answers_flag:
                            story_inputs.append(lexicons_dictionary[word.lower()])
                        else:
                            story_inputs.append(lexicons_dictionary['-'])
                            story_outputs.append(lexicons_dictionary[word.lower()])

                        # set the answers_flags if a question mark is encountered
                        if not answers_flag:
                            answers_flag = (word == '?')

        llprint("\rEncoding Data ... %d/%d" % (indx + 1, len(files_list)))

    print("\rEncoding Data ... Done!")
    return files, stories_lengths


def process_csv(path):
    data = []
    dictionary = {'<EOS>': 1}
    nlp = spacy.load('en')

    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['Language'] == 'English':
                for i in nlp(row['Description']):
                    try:
                        dictionary[str(i)] += 0
                    except:
                        dictionary[str(i)] = len(dictionary) + 1
                row['Description'] += ' <EOS>'
                data.append(row)
    random.shuffle(data)

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

    with open('MSR_en.csv', 'w') as enfile:
        writer = csv.DictWriter(enfile, keys)
        writer.writeheader()
        writer.writerows(data)

    with open('MSR_en_dict.csv', 'w') as dict_file:
        writer = csv.DictWriter(dict_file, ['word', 'id'])
        writer.writeheader()

        for key in dictionary.keys():
            writer.writerow({'word': key, 'id': dictionary[key]})
    return

if __name__ == '__main__':
    process_csv('./dataset/MSR.csv')

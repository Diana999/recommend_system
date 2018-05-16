import csv

import pandas as pd
from tqdm import tqdm

from structures.prediction_tree import PredictionTree


class CPTOriginal():
    alphabet = None  # A set of all unique items in the entire data file
    root = None  # Root node of the Prediction Tree
    II = None  # Inverted Index dictionary, where key : unique item, value : set of sequences containing this item
    LT = None  # A Lookup table dictionary, where key : id of a sequence(row), value: leaf node of a Prediction Tree

    def __init__(self):
        self.alphabet = set()
        self.root = PredictionTree()
        self.II = {}
        self.LT = {}

    def load_files(self, train_file, test_file, merge=False):

        train = []  # List of list containing the entire sequence data using which the model will be trained.
        test = []  # List of list containing the test sequences whose next n items are to be predicted

        if train_file is None:
            return train_file

        with open(train_file, 'rU') as f:
            train = list(list(map(int, rec)) for rec in csv.reader(f, delimiter=','))
        if test_file is None:
            return train, test_file

        with open(test_file, 'rU') as f:
            test = list(list(map(int, rec)) for rec in csv.reader(f, delimiter=','))
        if merge:
            train += test

        return train, test

    def train(self, train):

        cursornode = self.root

        for seqid, row in enumerate(train):
            for element in row:
                if cursornode.hasChild(element) == False:
                    cursornode.addChild(element)
                    cursornode = cursornode.getChild(element)
                else:
                    cursornode = cursornode.getChild(element)

                if self.II.get(element) is None:
                    self.II[element] = set()

                self.II[element].add(seqid)

                self.alphabet.add(element)

            self.LT[seqid] = cursornode

            cursornode = self.root

        return True

    def score(self, counttable, key, length, target_size, number_of_similar_sequences, number_items_counttable):


        weight_level = 1 / number_of_similar_sequences
        weight_distance = 1 / number_items_counttable
        score = 1 + weight_level + weight_distance * 0.001

        if counttable.get(key) is None:
            counttable[key] = score
        else:
            counttable[key] = score * counttable.get(key)

        return counttable

    def predict(self, train, test, k, n=1):

        predictions = []
        ttl = []
        for each_target in tqdm(test):
            ttl.append(each_target)
            each_target = each_target[-k:]
            intersection = set(range(0, len(train)))
            for element in each_target:
                if self.II.get(element) is None:
                    continue
                intersection = intersection & self.II.get(element)

            similar_sequences = []

            for element in intersection:
                currentnode = self.LT.get(element)
                tmp = []
                while currentnode.Item is not None:
                    tmp.append(currentnode.Item)
                    currentnode = currentnode.Parent
                similar_sequences.append(tmp)

            for sequence in similar_sequences:
                sequence.reverse()

            counttable = {}

            for sequence in similar_sequences:
                try:
                    index = next(
                        i for i, v in zip(range(len(sequence) - 1, 0, -1), reversed(sequence)) if v == each_target[-1])
                except:
                    index = None
                if index is not None:
                    count = 1
                    for element in sequence[index + 1:]:
                        if element in each_target:
                            continue

                        counttable = self.score(counttable, element, len(each_target), len(each_target),
                                                len(similar_sequences), count)
                        count += 1
            pred = self.get_n_largest(counttable, n)
            predictions.append(pred)

        return predictions, ttl

    def get_n_largest(self, dictionary, n):

        largest = sorted(dictionary.items(), key=lambda t: t[1], reverse=True)[:n]
        return [key for key, _ in largest]

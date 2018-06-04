import itertools
import math
import random
from collections import Counter

from tqdm import tqdm

from build_data.build_data import CPTMakeData
from src.structures.prediction_tree import PredictionTree


class CPT:
    alphabet = None  # A set of all unique items in the entire data file
    root = None  # Root node of the Prediction Tree
    II = None  # Inverted Index dictionary, where key : unique item, value : set of sequences containing this item
    LT = None  # A Lookup table dictionary, where key : id of a sequence(row), value: leaf node of a Prediction Tree

    def __init__(self):
        self.alphabet = set()
        self.root = PredictionTree()
        self.II = {}
        self.LT = {}

    def train(self, data, train, merge=False):
        print('trr')
        if merge:
            data += train
        cursornode = self.root

        for seqid, row in tqdm(enumerate(data)):
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

    def score(self, counttable, key, number_of_similar_sequences, number_items_counttable, num=None, sims=None,
              elem_count=None):

        weight_level = 1 / number_of_similar_sequences
        weight_distance = 1 / number_items_counttable
        score = 1 + weight_level * 5.5 + weight_distance * 1.5

        if counttable.get(key) is None:
            counttable[key] = score
        else:
            counttable[key] = score * counttable.get(key)

        return counttable

    def predict(self, data, target, k, n):
        predictions = []
        ttl = []
        for each_target in tqdm(target):
            ttl.append(each_target)
            each_target = each_target[math.ceil(len(each_target) * k):]
            intersection = set(range(0, len(data)))
            tries = []
            for element in each_target:
                if self.II.get(element) is None:
                    continue
                intersection = intersection & self.II.get(element)
                tries += list(self.II.get(element))
            similar_sequences = []
            if len(tries) > 1:
                intersection = [i[0] for i in Counter(tries).most_common(int(len(tries)))]
            else:
                intersection = tries
            # intersection = tries

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
                        count += 1
                        counttable = self.score(counttable, element, len(similar_sequences), count, ttl[-1],
                                                similar_sequences, self.II.get(element))

            pred = self.get_n_largest(counttable, n)
            predictions.append(pred)

        return predictions, ttl

    def get_n_largest(self, dictionary, n):
        largest = sorted(dictionary.items(), key=lambda t: t[1], reverse=True)[:n]
        return [key for key, _ in largest]


class CPTDummy(CPT):
    def predict(self, data, target, k, n):
        predictions = []
        ttl = []
        for each_target in tqdm(target):
            ttl.append(each_target)
            each_target = each_target[math.ceil(len(each_target) * k):]
            tries = []
            for element in each_target:
                if self.II.get(element) is None:
                    continue
                tries += list(self.II.get(element))
            similar_sequences = []
            if len(tries) > 1:
                intersection = [i[0] for i in Counter(tries).most_common(int(len(tries)))]
            else:
                intersection = tries

            for element in intersection:
                currentnode = self.LT.get(element)
                tmp = []
                while currentnode.Item is not None:
                    tmp.append(currentnode.Item)
                    currentnode = currentnode.Parent
                similar_sequences.append(tmp)

            similar_sequences = list(set(itertools.chain(*similar_sequences)))

            predictions.append([random.choice(similar_sequences if similar_sequences else [0]) for _ in range(n)])

        return predictions, ttl


class CPTFun(CPT):
    def __init__(self):
        super().__init__()
        self.items_genre = {}
        sequences = CPTMakeData().sequences
        self.sequnce_genre = {}
        with open("/Users/dianagajnutdinova/Codes/Python/PycharmProjects/untitled3/src/data/item_genre.txt", 'r') as f:
            for i in tqdm(f.readlines()):
                self.items_genre[i.split()[0]] = i.split()[1]
        for num, seq in tqdm(enumerate(sequences)):
            genres_list = []
            for item in seq:
                if self.items_genre.get(item):
                    genres_list.append(self.items_genre[item])
            self.sequnce_genre[num] = [i[0] for i in Counter(genres_list).most_common(1)]

    def score(self, counttable, key, number_of_similar_sequences, number_items_counttable, num=None, sims=None,
              elem_count=None):
        sims = 1 / len([1 for i in sims if key in i])
        score = 0
        if self.items_genre.get(key):
            if self.items_genre.get(key) in [i[0] for i in
                                             Counter([self.items_genre.get(i, 0) for i in num]).most_common(1)]:
                score *= score
            else:
                score *= score * 0.5
        else:
            score *= score * 0.5
        score += sims * 0.5 + len(elem_count)

        if counttable.get(key) is None:
            counttable[key] = score
        else:
            counttable[key] = score + counttable.get(key)
        return counttable

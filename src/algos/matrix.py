import itertools

import numpy as np
import pandas as pd
from tqdm import tqdm

from implemnts.cpt_original import CPTMakeData


class CreateUserBookMatrix():
    def __init__(self):
        self._sequence_reader()
        self._build_target()

    def _sequence_reader(self):
        self.book_sequences = CPTMakeData('a', num_of_seq=5000).sequences
        self.num_users = len(self.book_sequences)
        print(self.num_users)
        self.unique_book_list = list(set(itertools.chain(*self.book_sequences)))
        print(len(self.unique_book_list))

    def _build_target(self):
        self.new_book_encoded = [sequence[:-1] for sequence in self.book_sequences]
        self.target_dict = {sequence[-1:][0]: sequence for sequence in self.book_sequences}

    def flatten_weigths(self, n):
        weigths = [5]
        for i in range(1, n):
            weigths.append(weigths[i - 1] * 0.8)
        return weigths[::-1]

    def _find_shoots(self, a) -> list:
        b = self.unique_book_list
        lst = iter(self.flatten_weigths(len(a)))
        diffs = set(b) - set(a)
        return [next(lst) if _ not in diffs else 0 for _ in b]

    def create_matrix(self):
        print("suffering")
        self.matrix = (np.array(list((map(lambda x: self._find_shoots(x), tqdm(self.new_book_encoded))))))
        return self.matrix.astype(float)


smth = CreateUserBookMatrix()
matrix = smth.create_matrix()
items = []
users = []
ratings = []

for i, user in enumerate(tqdm(matrix)):
    users += [i for i in range(len(smth.book_sequences[i]))]
    items += smth.book_sequences[i]
    ratings += smth.flatten_weigths(len(smth.book_sequences[i]))
from collections import defaultdict

from surprise import SVD
from surprise import Dataset
from surprise import Reader


def get_top_n(predictions, n=10):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


ratings_dict = {'itemID': items,
                'userID': users,
                'rating': ratings}
df = pd.DataFrame(ratings_dict)

reader = Reader(rating_scale=(0, 5))

data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)

trainset = data.build_full_trainset()
print("> Training...")
algo = SVD()
algo.train(trainset)
print("> OK")
testset = trainset.build_anti_testset()
predictions = algo.test(testset)

top_n = get_top_n(predictions, n=10)
print("> OK")
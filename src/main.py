import itertools
import operator
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import train_test_split
from surprise.prediction_algorithms.matrix_factorization import SVDpp
from tqdm import tqdm

from build_data.build_data_matrix import MatrixMakeData


def get_top_n(predictions, n=3):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


class CreateUserBookMatrix:
    def __init__(self):
        self._build_target()

    def _build_target(self):
        data = MatrixMakeData(num_of_seq=350)
        data.make_cut_off()  # splitting in 80/20 test/target
        self.test, self.train = data.test, data.train
        self.new_book_encoded = self.train
        self.unique_book_list = list(set(itertools.chain(*(self.new_book_encoded + self.test))))

    def flatten_weigths(self, n):
        weigths = [5]
        for i in range(1, n):
            weigths.append((weigths[i - 1] * 0.89))
        return weigths[::-1]


smth = CreateUserBookMatrix()


def prepare_df(listik):
    items = []
    users = []
    ratings = []
    for i, user in enumerate(tqdm(smth.new_book_encoded)):
        users += [i for _ in range(len(listik[i]))]
        items += listik[i]
        ratings += smth.flatten_weigths(len(listik[i]))
    ratings_dict = {'itemID': items,
                    'userID': users,
                    'rating': ratings}
    df = pd.DataFrame(ratings_dict)
    reader = Reader(rating_scale=(0, 5))
    random.shuffle(Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader).raw_ratings)
    return Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)

# нужно обрезать все последовательности и потом предсказать предмет с наибольшей оценкой

data_train = prepare_df(smth.new_book_encoded)
data_trains = data_train.build_full_trainset()
print("> Training...")
algo = SVDpp()
algo.fit(data_trains)
print("> OK")
supra = 0
predictions = algo.test(data_trains.build_testset())
top_n = get_top_n(predictions, n=20)
j = 0
for i in range(len(top_n)):
    if len(set(smth.test[i]) & set([i[0] for i in top_n[i]])):
        supra += 1
j = 0

import itertools
from collections import defaultdict, Counter

import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise import SVD
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
        self.items_genre = {}
        sequences = data.sequences
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

    def flatten_weigths(self, n, arr):
        weigths = [5]
        for i in range(1, n):
            weigths.append((weigths[i - 1] * 0.89))
        return weigths

    def flatten_w_s(self, n, arr):
        weigths = [4.5]
        for i in range(1, n):
            weigths.append((weigths[i - 1] * 0.89))
        for j in range(len(arr)):
            if self.items_genre.get(arr[j]):

                if self.items_genre.get(arr[j]) in [i[0] for i in
                                                    Counter([self.items_genre.get(i, 0) for i in arr]).most_common(1)]:
                    weigths[j] = 5
        return weigths


def prepare_df_train(smth, listik):
    items = []
    users = []
    ratings = []
    for i, user in enumerate(tqdm(smth.new_book_encoded)):
        users += [i for _ in range(len(listik[i]))]
        items += listik[i]
        ratings += smth.flatten_weigths(len(listik[i]), listik[i])
    ratings_dict = {'itemID': items,
                    'userID': users,
                    'rating': ratings}
    df = pd.DataFrame(ratings_dict)
    reader = Reader(rating_scale=(1, 5))
    return Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)


def create_work():
    smth = CreateUserBookMatrix()
    data_train = prepare_df_train(smth, smth.new_book_encoded)
    data_trains = data_train.build_full_trainset()
    k = data_trains.build_anti_testset()

    print("> Training...")
    algo = SVD()
    algo.fit(data_trains)
    print("> OK")
    supra = 0
    predictions = algo.test(k)
    top_n = get_top_n(predictions, n=25)
    j = 0
    for i in range(len(top_n)):
        if len(set(smth.test[i]) & set([i[0] for i in top_n[i]])):
            supra += 1
    j = 0
    print(supra)

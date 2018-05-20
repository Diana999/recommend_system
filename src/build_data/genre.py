import itertools
from collections import Counter

import pandas as pd
from tqdm import tqdm

from build_data.make_data import MakeData


def get_items_genre_dict():
    item_genre_dict = {}
    df = pd.read_csv("data/book_genre_encoded.csv")
    for i in tqdm(range(len(df))):
        item_genre_dict[df.iloc[i][0]] = df.iloc[i][1]
    with open('item_genre.txt', 'w') as f:
        for i,j in item_genre_dict.items():
            f.write(str(i) + ' ' + str(j) + '\n')
    return item_genre_dict


def get_sequence_most_n_genre(n):
    data = MakeData()
    sequences = data.sequences_from_file()
    seq_genre = {}
    df = pd.read_csv("data/book_genre_encoded.csv")
    for num, seq in tqdm(enumerate(sequences)):
        genres_list = list(df[df['a'].isin(seq)]['b'])
        most_common = [i[0] for i in Counter(genres_list).most_common(n)]
        seq_genre[num] = most_common
    with open('sequence_genre.txt', 'w') as f:
        for seq, item in tqdm(seq_genre.items()):
            f.write(str(seq) + ' ')
            for genre in item:
                f.write(str(genre) + ' ')
            f.write('\n')
    return seq_genre

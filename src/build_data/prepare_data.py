import itertools

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def delete_short_sequences():
    with open('data/whole_data.txt', 'r') as f:
        with open('data/data_with_len_more_2.txt', 'w') as d:
            for line in tqdm(f.readlines()):
                line_split = line.split()
                if len(line_split) > 2:
                    d.write(line)


def encode_data():
    sequences = []
    with open('data/normal_whole_data.txt', 'r') as f:
        print("Reading...")
        for line in tqdm(f.readlines()):
            sequences.append(line.split())
    print("Encoding...")
    encoder = {}
    for num, i in tqdm(enumerate(list(set(itertools.chain(*sequences))))):
        encoder[i] = num
    print("Learn how to encode...")
    with open("encodes.txt", 'w') as f:
        for i, j in encoder.items():
            f.write(str(i) + ' : ' + str(j) + '\n')

    sequences = [[encoder[i] for i in j] for j in tqdm(sequences)]
    with open('encoded_data_whole_normal.txt', 'w') as f:
        f.writelines([' '.join(list(map(str, i))) + '\n' for i in tqdm(sequences)])


class PreprocessDf():
    def __init__(self):
        self.df = pd.read_csv('/Users/dianagajnutdinova/Downloads/book_ratings_encoding.csv', index_col=False,
                              delimiter=':::')

    def start(self):
        self.user_map = self.encode('user_id')
        self.book_map = self.encode('book_id')
        self.authors_map = self.encode('authors')
        return self.df, self.user_map, self.book_map

    def encode(self, name):
        label = LabelEncoder()
        label.fit(self.df[name].drop_duplicates())  # задаем список значений для кодирования
        self.df[name] = label.transform(self.df[name])
        return list(label.classes_)

    def find_length_of_sequence(self):
        print(len(self.df.user_id.unique()))
        for num, i in self.df.user_id.unique():
            if not self.find_values_for_user(i):
                self.df.drop(self.df[self.df['user_id'] == i].index)
            if num % 1000 == 0:
                print(num)
        return self.df

    def find_values_for_user(self, user_id):
        length = len([self.df[self.df['user_id'] == user_id]])
        return True if length == 1 else False

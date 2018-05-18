import os

from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


class MakeData:
    def __init__(self, len_of_seq_fixed=None, len_of_seq=None, num_of_seq=None):
        self.length_of_seq_fixed = len_of_seq_fixed
        self.len_of_seq = len_of_seq
        self.num_of_seq = num_of_seq
        self.sequences = []

    def make_files(self):
        print('making files')
        self.split_file_into_n_files('data/whole_data.txt', 1)

    def sequences_from_file(self, file=None):
        if not file:
            file = 'data/encoded_data_len_more_2.txt'
        with open(file, 'r') as f:
            for line in tqdm(f.readlines()):
                self.sequences.append(line.split())
        if self.len_of_seq:
            self.sequences = [seq for seq in self.sequences if len(seq) > self.len_of_seq]
        elif self.length_of_seq_fixed:
            self.sequences = [seq for seq in self.sequences if len(seq) == self.length_of_seq_fixed]
        if self.num_of_seq:
            self.sequences = self.sequences[:self.num_of_seq]
        self.sequences = [list(map(int, i)) for i in self.sequences]
        # scaler = MinMaxScaler()
        # scaler.fit(self.sequences)
        # self.sequences = scaler.transform(self.sequences)
        return self.sequences

    def make_sequences(self):
        print('we')
        train_seq, test_seq_begin = self.split_data_to_train_test(self.sequences, 0.9)
        test_seq, self.target_test_seq = self.split_list_of_seq_into_test_and_target(test_seq_begin)
        return self.target_test_seq, test_seq, train_seq

    def split_file_into_test_and_target(self, file):
        with open(file, 'r') as f:
            with open('without_target.txt', 'w') as without_target:
                with open('target.txt', 'w') as target:
                    for line in f.readlines():
                        line = line.split()
                        if len(line) < 2:
                            continue
                        else:
                            without_target.write(' '.join(line[:-1]))
                            target.write(' '.join(line[-1:]))

    def split_list_of_seq_into_test_and_target(self, list_of_seq):
        return [seq[:-1] for seq in list_of_seq], [seq[-1:][0] for seq in list_of_seq]

    def split_file_into_n_files(self, file, n):
        for i in range(1, 100):
            try:
                os.remove("data/{}_part.txt".format(i))
            except:
                pass
        mapping = {i: 'data/{}_part.txt'.format(i) for i in range(1, n + 1)}
        c = 1
        with open(file, 'r') as f:
            for line in f.readlines():
                with open(mapping[c], 'a') as file:
                    file.write(line)
                c += 1
                if c > n:
                    c = 1

    def split_data_to_train_test(self, array, proportion):
        return array[0:int(len(array) * proportion)], array[int(len(array) * proportion):]

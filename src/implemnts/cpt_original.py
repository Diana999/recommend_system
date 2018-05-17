from tqdm import tqdm

from build_data.make_target import split_list_of_seq_into_test_and_target
from build_data.split_data import split_data_to_train_test, split_file_into_n_files


class CPTMakeData:
    def __init__(self, file, length_of_seq_fixed=None, len_of_seq=None, num_of_seq=None):
        self.length_of_seq_fixed = length_of_seq_fixed
        self.len_of_seq = len_of_seq
        self.num_of_seq = num_of_seq
        self.sequences = []
        self.read_file(file)

    def make_files(self):
        print('making files')
        split_file_into_n_files('data/whole_data.txt', 1)

    def read_file(self, file):
        with open('data/whole_data.txt', 'r') as f:
            for line in tqdm(f.readlines()):
                self.sequences.append(line.split())
        if self.len_of_seq:
            self.sequences = [seq for seq in self.sequences if len(seq) > self.len_of_seq]
        elif self.length_of_seq_fixed:
            self.sequences = [seq for seq in self.sequences if len(seq) == self.length_of_seq_fixed]
        if self.num_of_seq:
            self.sequences = self.sequences[:self.num_of_seq]

    def make_sequences(self):
        print('we')
        train_seq, test_seq_begin = split_data_to_train_test(self.sequences, 0.9)
        test_seq, self.target_test_seq = split_list_of_seq_into_test_and_target(test_seq_begin)
        return self.target_test_seq, test_seq, train_seq

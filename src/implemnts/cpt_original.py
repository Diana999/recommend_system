from tqdm import tqdm

from build_data.make_target import split_list_of_seq_into_test_and_target
from build_data.split_data import split_data_to_train_test, split_file_into_n_files


class CPTOriginalWork:
    def __init__(self, file, length, amount):
        self.sequences = []
        # self.make_files()
        self.read_file(file, length, amount)

    def make_files(self):
        print('making files')
        split_file_into_n_files('data/whole_data.txt', 1)

    def read_file(self, file, length, amount):
        with open('data/whole_data.txt', 'r') as f:
            print("Reading...")
            k = 0
            for line in tqdm(f.readlines()):
                if len(line.split()) > length and k < amount:
                    self.sequences.append(line.split())
                    k += 1

    def make_sequences(self):
        print('we')
        train_seq, test_seq_begin = split_data_to_train_test(self.sequences, 0.9)
        test_seq, self.target_test_seq = split_list_of_seq_into_test_and_target(test_seq_begin)
        return self.target_test_seq, test_seq, train_seq

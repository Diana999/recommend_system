from tqdm import tqdm


def split_data_to_train_test(array, proportion):
    return array[0:int(len(array) * proportion)], array[int(len(array) * proportion):]


def split_list_of_seq_into_test_and_target(list_of_seq):
    return [seq[:-1] for seq in list_of_seq], [seq[-1:][0] for seq in list_of_seq]


class CPTMakeData:
    def __init__(self, file, length_of_seq_fixed=None, len_of_seq=None, num_of_seq=None):
        self.length_of_seq_fixed = length_of_seq_fixed
        self.len_of_seq = len_of_seq
        self.num_of_seq = num_of_seq
        self.sequences = []
        self.read_file(file)

    def read_file(self, file):
        with open('data/data_with_len_more_2.txt', 'r') as f:
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
        d = 1
        return self.target_test_seq, test_seq, train_seq

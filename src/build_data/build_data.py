from tqdm import tqdm


def split_list_of_seq_into_test_and_target(list_of_seq, prop):
    return [seq[:-1] for seq in list_of_seq], [seq[len(seq) * prop:] for seq in list_of_seq]


class CPTMakeData:
    def __init__(self, length_of_seq_fixed=None, len_of_seq=None, num_of_seq=None):
        self.length_of_seq_fixed = length_of_seq_fixed
        self.len_of_seq = len_of_seq
        self.num_of_seq = num_of_seq
        self.sequences = []
        self.read_file()
        self.make_train_test()
        self.make_test_suffix_preffix()

    def read_file(self, ):
        with open('data/encoded_data_whole_normal_less_50.txt', 'r') as f:
            for line in tqdm(f.readlines()):
                self.sequences.append(line.split())
        if self.len_of_seq:
            self.sequences = [seq for seq in self.sequences if len(seq) > 10 and len(seq) < 50]
        elif self.length_of_seq_fixed:
            self.sequences = [seq for seq in self.sequences if len(seq) == self.length_of_seq_fixed]
        if self.num_of_seq:
            self.sequences = self.sequences[:self.num_of_seq]

    @staticmethod
    def split_data_with_proportion(arr, proportion):
        return arr[0:int(len(arr) * proportion)], arr[int(len(arr) * proportion):]

    @staticmethod
    def split_list_of_seq_into_test_and_target(list_of_seq, prop):
        return [seq[int(len(seq) * prop):] for seq in list_of_seq], [seq[:int(len(seq) * prop)] for seq in list_of_seq]

    def make_train_test(self):
        self.train, self.test = self.split_data_with_proportion(self.sequences, 0.9)

    def make_test_suffix_preffix(self):
        self.test_suffix, self.test_preffix = self.split_list_of_seq_into_test_and_target(self.test, 0.7)

    def make_sequences(self):
        print('we')
        train_seq, test_seq_begin = self.split_data_with_proportion(self.sequences, 0.9)
        test_seq, self.target_test_seq = self.split_list_of_seq_into_test_and_target(test_seq_begin, 0.8)
        return self.target_test_seq, test_seq, train_seq

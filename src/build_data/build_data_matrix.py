from build_data.build_data import CPTMakeData


class MatrixMakeData(CPTMakeData):
    def __init__(self, length_of_seq_fixed=None, len_of_seq=None, num_of_seq=None):
        super().__init__(length_of_seq_fixed=length_of_seq_fixed, len_of_seq=len_of_seq, num_of_seq=num_of_seq)

    def make_train_test(self):
        pass

    def make_test_suffix_preffix(self):
        pass

    def make_cut_off(self):
        self.test, self.train  = self.split_list_of_seq_into_test_and_target(self.sequences, 0.8)
        g = 0

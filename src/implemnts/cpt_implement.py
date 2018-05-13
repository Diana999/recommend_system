import itertools

from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from algos.cpt_improved import CPTImproved
from src.build_data.make_target import split_list_of_seq_into_test_and_target
from src.build_data.split_data import split_data_to_train_test, split_file_into_n_files


class CPTWork:
    def __init__(self, file):
        self.sequences = []
        self.make_files()
        self.read_file(file)
        self.encode_data()
        self.predict()

    def make_files(self):
        print('making files')
        split_file_into_n_files('data/data_with_len_more_2.txt', 10)

    def read_file(self, file):
        with open(file, 'r') as f:
            print("Reading...")
            for line in tqdm(f.readlines()):
                if len(line.split()) > 7:
                    self.sequences.append(line.split())

    def encode_data(self):
        print("Encoding...")
        label = LabelEncoder()
        self.unique_books = list(set(itertools.chain(*self.sequences)))
        label.fit(self.unique_books)
        print("Learn how to encode...")
        self.sequences = [list(label.transform(j)) for j in tqdm(self.sequences)]

    def predict(self):
        print("Writing...")
        train_seq, test_seq = split_data_to_train_test(self.sequences, 0.9)
        test_seq, self.target_test_seq = split_list_of_seq_into_test_and_target(test_seq)
        print("Prediction...")
        model = CPTImproved(train_seq, test_seq)
        model.train()
        predictions, ttl = model.predict(test_seq, 2, 1)
        with open('data/show.txt', 'w') as f:
            for i in range(len(predictions)):
                f.write(str(predictions[i]) + ' ' + str(self.target_test_seq[i]) + '\n')
                a = predictions[i][0] if predictions[i] else None
                b = self.target_test_seq[i][0]
                if a == b:
                    f.write("YES")

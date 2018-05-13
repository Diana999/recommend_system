import csv
import itertools

from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from algos.cpt import CPT
from src.build_data.make_target import split_list_of_seq_into_test_and_target
from src.build_data.split_data import split_data_to_train_test, split_file_into_n_files


class CPTWorkFixed:
    def __init__(self, file, len_of_sequence):
        self.sequences = []
        self.make_files()
        self.read_file(file,len_of_sequence)
        self.encode_data()
        #self.predict(len_of_tail)

    def make_files(self):
        print('making files')
        split_file_into_n_files('data/data_with_len_more_2.txt', 25)

    def read_file(self, file,len_of_sequence):
        with open(file, 'r') as f:
            print("Reading...")
            for line in tqdm(f.readlines()):
                if len(line.split()) > len_of_sequence:
                    self.sequences.append(line.split())

    def encode_data(self):
        print("Encoding...")
        label = LabelEncoder()
        self.unique_books = list(set(itertools.chain(*self.sequences)))
        label.fit(self.unique_books)
        print("Learn how to encode...")
        self.sequences = [list(label.transform(j)) for j in tqdm(self.sequences)]

    def predict(self, n):
        print("Writing...")
        train_seq, test_seq_begin = split_data_to_train_test(self.sequences, 0.9)
        d = 1
        test_seq, self.target_test_seq = split_list_of_seq_into_test_and_target(test_seq_begin)
        with open('data/test.csv', 'w') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            for i in test_seq:
                wr.writerow(i)
        with open('data/train.csv', 'w') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            for i in train_seq:
                wr.writerow(i)
        print("Prediction...")
        model = CPT()
        model.train(train_seq+test_seq)
        predictions, ttl = model.predict(train_seq + test_seq, test_seq, n, 1)
        counter_good = 0
        with open('data/show.txt', 'w') as f:
            for i in range(len(predictions)):
                f.write(str(predictions[i]) + ' ' + str(self.target_test_seq[i]) + '\n')
                a = predictions[i][0] if predictions[i] else None
                b = self.target_test_seq[i][0]
                if a == b:
                    f.write("YES")
                    counter_good += 1
        print(predictions[0] if predictions[0] else None, ttl[0], test_seq[0], self.target_test_seq[0])
        return float(counter_good / len(predictions))

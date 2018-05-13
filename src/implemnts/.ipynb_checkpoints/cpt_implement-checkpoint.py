import csv
import itertools

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from algos.cpt import CPT
from src.build_data.make_target import split_list_of_seq_into_test_and_target
from src.build_data.split_data import split_data_to_train_test


class CPTWork:
    def __init__(self):
        self.sequences = []

        self.read_file()
        self.encode_data()
        self.scale_array(self.sequences)
        self.write_files()
        self.predict()

    def read_file(self, file):
        with open(file, 'r') as f:
            for line in f.readlines():
                self.sequences.append(line.split())

    def encode_data(self):
        label = LabelEncoder()
        label.fit(list(set(itertools.chain(*self.sequences))))
        self.sequences = [label.transform(i) for i in self.sequences]

    def scale_array(self, array):
        scaler = MinMaxScaler()
        scaler.fit(array)
        return scaler.transform(array)

    def write_files(self):
        train_seq, test_seq = split_data_to_train_test(self.sequences, 0.9)
        test_seq, self.target_test_seq = split_list_of_seq_into_test_and_target(test_seq)
        with open('data/test.csv', 'w') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            for i in test_seq:
                wr.writerow(i)
        with open('data/train.csv', 'w') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            for i in train_seq:
                wr.writerow(i)

    def predict(self):
        model = CPT()
        data, test = model.load_files("data/train.csv", "data/test.csv")
        model.train(data)
        predictions, ttl = model.predict(data, test, 2, 1)
        with open('data/show.txt', 'w') as f:
            for i in range(len(predictions)):
                f.write(str(predictions[i]) + ' ' + str(self.target_test_seq[i]))

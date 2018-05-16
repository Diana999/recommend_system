import csv
import itertools

from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from algos.cpt_original import CPTOriginal
from build_data.make_target import split_list_of_seq_into_test_and_target
from build_data.split_data import split_data_to_train_test, split_file_into_n_files


class CPTOriginalWork:
    def __init__(self, file, length, amount):
        self.sequences = []
        # self.make_files()
        self.read_file(file, length, amount)
        # self.encode_data()

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

    def encode_data(self):
        print("Encoding...")
        label = LabelEncoder()
        self.unique_books = list(set(itertools.chain(*self.sequences)))
        label.fit(self.unique_books)
        print("Learn how to encode...")
        self.sequences = [list(label.transform(j)) for j in tqdm(self.sequences)]

    def predict(self):
        print('we')
        train_seq, test_seq_begin = split_data_to_train_test(self.sequences, 0.9)
        test_seq, self.target_test_seq = split_list_of_seq_into_test_and_target(test_seq_begin)
        # with open('data/test.csv', 'w') as myfile:
        #     wr = csv.writer(myfile, quoting=csv.QUOTE_NONE)
        #     for i in test_seq:
        #         wr.writerow(i)
        # with open('data/train.csv', 'w') as myfile:
        #     wr = csv.writer(myfile, quoting=csv.QUOTE_NONE)
        #     for i in train_seq:
        #         wr.writerow(i)
        # print("Prediction...")
        return self.target_test_seq, test_seq, train_seq
        # model = CPTOriginal()
        # train, test = model.load_files("data/train.csv", "data/test.csv", merge=True)
        # model.train(train)
        # predictions, ttl = model.predict(train, test, n, 1)
        # counter_good = 0
        # with open('data/show.txt', 'w') as f:
        #     for i in range(len(predictions)):
        #         f.write(str(predictions[i]) + ' ' + str(self.target_test_seq[i]) + '\n')
        #         a = predictions[i][0] if predictions[i] else None
        #         b = self.target_test_seq[i][0]
        #         if a == b:
        #             f.write("YES")
        #             counter_good += 1
        # #return(float(len([1 for i in predictions if i ])/len(predictions)))
        # print(float(len([1 for i in predictions if i]) / len(predictions)))
        # print(float(counter_good / len(predictions)))

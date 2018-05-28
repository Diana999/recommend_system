import itertools

import numpy as np
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import SimpleRNN
from keras.models import Sequential
from tqdm import tqdm

from build_data.make_data import MakeData


class BuildNeuralLSTM():
    def __init__(self, length_of_seg=None, length_of_seq_fixed=6, num_of_seq=700, hidden_layers=512, batch_size=128,
                 epochs=10):
        self.length_of_seg = length_of_seg
        self.length_of_seq_fixed = length_of_seq_fixed
        self.num_of_seq = num_of_seq
        self.hidden_layers = hidden_layers
        self.batch_size = batch_size
        self.epochs = epochs

    def make_data(self):
        smth = MakeData(len_of_seq_fixed=self.length_of_seq_fixed, num_of_seq=self.num_of_seq)
        self.sequences = smth.sequences_from_file()
        char_to_index = {v: i for i, v in enumerate((list(set(itertools.chain(*self.sequences)))))}
        self.index_to_char = {i: v for i, v in enumerate((list(set(itertools.chain(*self.sequences)))))}
        self.sequences = [[char_to_index[i] for i in j] for j in tqdm(self.sequences)]
        self.train, self.test = smth.split_data_to_train_test(self.sequences, 0.9)
        self.test_begin, self.test_end = smth.split_list_of_seq_into_test_and_target(self.test)
        self.train_begin, self.train_end = smth.split_list_of_seq_into_test_and_target(self.train)

    def prepare_model(self):
        vocab_size = max(list(set(itertools.chain(*self.train)))) + 2

        self.model = Sequential([
            Embedding(vocab_size, len(self.test_begin), input_length=self.length_of_seq_fixed - 1),  # а так нужно?
            SimpleRNN(self.hidden_layers, activation='relu'),
            Dense(vocab_size, activation='softmax')
        ])

        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    def fit_model(self):
        self.model.fit(np.array(self.train_begin), np.array(self.train_end), batch_size=self.batch_size,
                       epochs=self.epochs)

    def predict_next_charpredict_(self, inp, num_of_pred=10):
        arr = np.expand_dims(np.array(inp), axis=0)
        prediction = self.model.predict(arr)
        u = [[i for i in j] for j in prediction][0]
        return [self.index_to_char[i] for i in np.array(u).argsort()[-num_of_pred:][::-1]]

    def gain_results(self):
        counter = 0
        for i in range(len(self.test_begin)):
            if self.index_to_char[self.test_end[i]] in self.predict_next_charpredict_(self.test_begin[i]):
                counter += 1
        return counter / len(self.test_begin)

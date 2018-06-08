import keras

from build_data.build_data import CPTMakeData
from build_data.build_data_matrix import MatrixMakeData
from implemnts.rnn import make_neural_work
import itertools

import numpy as np
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import SimpleRNN
from keras.models import Sequential
from tqdm import tqdm



class BuildNeuralLSTM():
    def __init__(self, num_of_seq=100, hidden_layers=512, batch_size=128,
                 epochs=10):
        self.num_of_seq = num_of_seq
        self.hidden_layers = hidden_layers
        self.batch_size = batch_size
        self.epochs = epochs

    def make_data(self):
        prop = 0.7

        smth = MatrixMakeData(num_of_seq=550)
        self.sequences = smth.sequences
        char_to_index = {v: i for i, v in enumerate((list(set(itertools.chain(*self.sequences)))))}
        self.index_to_char = {i: v for i, v in enumerate((list(set(itertools.chain(*self.sequences)))))}
        self.sequences = [[char_to_index[i] for i in j] for j in tqdm(self.sequences)]
        self.train, self.test = [seq[int(len(seq) * prop):] for seq in self.sequences], [seq[:int(len(seq) * prop)] for
                                                                                         seq in self.sequences]
        self.test_begin, self.test_end = [seq[int(len(seq) * prop):] for seq in self.test], [seq[:int(len(seq) * prop)]
                                                                                             for seq in self.test]
        self.train_begin, self.train_end = [seq[:-1] for seq in self.train], [seq[-1:][0] for seq in self.train]


    def prepare_model(self):
        vocab_size = max(list(set(itertools.chain(*self.train)))) + 2
        self.a = keras.preprocessing.sequence.pad_sequences(np.array(self.train_begin),
                                                            maxlen=50)

        self.model = Sequential([
            Embedding(vocab_size, len(self.test_begin), input_length=50),  # а так нужно?
            SimpleRNN(self.hidden_layers, activation='relu'),
            Dense(vocab_size, activation='softmax')
        ])

        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    def fit_model(self):
        self.model.fit(self.a, np.array(self.train_end), batch_size=self.batch_size,
                       epochs=self.epochs)

    def predict_next_charpredict_(self, inp, num_of_pred=20):
        arr = np.expand_dims(np.array(inp), axis=0)
        prediction = self.model.predict(arr)
        u = [[i for i in j] for j in prediction][0]
        return [self.index_to_char[i] for i in np.array(u).argsort()[-num_of_pred:][::-1]]

    def gain_results(self):
        counter = 0
        self.am = keras.preprocessing.sequence.pad_sequences(np.array(self.test_begin),
                                                            maxlen=50)
        for i in range(len(self.test_begin)):
            if len(set([self.index_to_char[j] for j in self.test_end[i]]) & set(self.predict_next_charpredict_(self.am[i]))):
                counter += 1
        return counter / len(self.test_begin)


smth = BuildNeuralLSTM()
smth.make_data()
smth.prepare_model()
smth.fit_model()
print(smth.gain_results())

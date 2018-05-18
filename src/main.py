import itertools
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import SimpleRNN
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
from tqdm import tqdm

from build_data.make_data import MakeData

smth = MakeData(len_of_seq_fixed=6, num_of_seq=1000)
sequences = smth.sequences_from_file()
char_to_index = {v: i for i, v in enumerate((list(set(itertools.chain(*sequences)))))}
index_to_char = {i: v for i, v in enumerate((list(set(itertools.chain(*sequences)))))}
sequences = [[char_to_index[i] for i in j] for j in tqdm(sequences)]
train, test = smth.split_data_to_train_test(sequences, 0.9)
test_begin, test_end = smth.split_list_of_seq_into_test_and_target(test)
train_begin, train_end = smth.split_list_of_seq_into_test_and_target(train)
hidden_layers = 256
vocab_size = max(list(set(itertools.chain(*train)))) + 1
print(vocab_size)  # 18940

model = Sequential([
    Embedding(vocab_size, 900, input_length=5),
    SimpleRNN(hidden_layers, activation='relu'),
    Dense(vocab_size, activation='softmax')
])
model.summary()
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam())
model.fit(np.array(train_begin), np.array(train_end), batch_size=64, epochs=5)
model.save_weights('simpleRNN_3pred.h5')
model.load_weights('simpleRNN_3pred.h5')
model.save_weights('simpleRNN_7pred.h5')
model.load_weights('simpleRNN_7pred.h5')


def predict_next_charpredict_(inp):
    index = [char_to_index[i] for i in inp]
    arr = np.expand_dims(np.array(index), axis=0)
    prediction = model.predict(arr)
    return index_to_char[np.argmax(prediction)]
d = 1

import itertools
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import SimpleRNN
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
from tqdm import tqdm


smth = MakeData(len_of_seq_fixed=6, num_of_seq=7000)
sequences = smth.sequences_from_file()
char_to_index = {v: i for i, v in enumerate((list(set(itertools.chain(*sequences)))))}
index_to_char = {i: v for i, v in enumerate((list(set(itertools.chain(*sequences)))))}
sequences = [[char_to_index[i] for i in j] for j in tqdm(sequences)]
train, test = smth.split_data_to_train_test(sequences, 0.9)
test_begin, test_end = smth.split_list_of_seq_into_test_and_target(test)
train_begin, train_end = smth.split_list_of_seq_into_test_and_target(train)
hidden_layers = 512
vocab_size = max(list(set(itertools.chain(*train)))) + 1
print(vocab_size)  # 18940

model = Sequential([
    Embedding(vocab_size, len(test_begin), input_length=5),
    SimpleRNN(hidden_layers, activation='relu'),
    Dense(vocab_size, activation='softmax')
])
model.summary()
model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
model.fit(np.array(train_begin), np.array(train_end), batch_size=128, epochs=5)
model.save_weights('simpleRNN_3pred.h5')
model.load_weights('simpleRNN_3pred.h5')
model.save_weights('simpleRNN_7pred.h5')
model.load_weights('simpleRNN_7pred.h5')
def predict_next_charpredict_(inp):
    arr = np.expand_dims(np.array(inp), axis=0)
    prediction = model.predict(arr)
    u = [[i for i in j] for j in prediction][0]
    return [index_to_char[i] for i in np.array(u).argsort()[-10:][::-1]]
    #return (index_to_char[np.argmax(prediction)])
predict_next_charpredict_(test_begin[0])
c = 0
for i in range(len(test_begin)):
    if (index_to_char[test_end[i]] in predict_next_charpredict_(test_begin[i])):
        c += 1
print(c/len(test_begin))
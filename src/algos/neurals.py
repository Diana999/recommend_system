import itertools

import numpy as np
from pybrain.datasets.sequential import SequentialDataSet
from pybrain.structure import SigmoidLayer, LSTMLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork

from build_data.make_data import MakeData

d = 1


def find_nearest(listed, element):
    mini = 100000000
    target_elem = -1
    for i in list(set(itertools.chain(*listed))):
        if abs(i - element) < mini:
            mini = abs(i - element)
            target_elem = i
    return(target_elem)


def begin():
    smth = MakeData(len_of_seq_fixed=6, num_of_seq=10000)
    sequences = smth.sequences_from_file()
    train, test = smth.split_data_to_train_test(sequences, 0.9)
    test_begin, test_end = smth.split_list_of_seq_into_test_and_target(test)
    train_begin, train_end = smth.split_list_of_seq_into_test_and_target(train)

    dataModel = [[tuple(begin), (end,)] for begin, end in zip(train_begin, train_end)]
    ds = SequentialDataSet(5, 1)
    for input, target in dataModel:
        ds.addSample(input, target)
    print(len(ds))
    net = buildNetwork(5, 12, 1, hiddenclass=LSTMLayer, outclass=SigmoidLayer, recurrent=True, bias=True)
    net.randomize()
    trainer = BackpropTrainer(net, ds, learningrate=0.00005)
    # trainer.trainUntilConvergence()#verbose=True,
    # dataset=ds,
    # maxEpochs=100)#validationData=ds,)
    # trainer.trainEpochs(1000)
    for _ in range(1000):
        print(trainer.train(),_)
    with open('show.txt', 'w') as f:
        for i, j in zip(test_begin, test_end):
            acti = net.activate(i)
            lol = j[0]
            f.write(str(acti) + ' ; ' + str(lol) + ' ' + str(find_nearest(sequences, acti)) + '\n')
            if abs(lol - acti) < 0.00001:
                f.write("yes")

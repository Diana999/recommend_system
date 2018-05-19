import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from algos.cpt import CPT
from implemnts.cpt_original import CPTMakeData


class CPTPictures:
    def __init__(self):
        pass

    def get_results_num_of_seq(self):
        self.length_list = []
        self.supra_list = []
        for num in tqdm(range(5000, 100000, 10000)):
            target = CPTMakeData(file='data/1_part.txt', len_of_seq=15, num_of_seq=num)
            target_test_seq, test_seq, train_seq = target.make_sequences()
            model = CPT()
            model.train(train_seq, test_seq, merge=False)
            predictions = model.predict(train_seq, test_seq, 5, 5)
            supra = 0
            print(len(target_test_seq))
            for ia in range(len(target_test_seq)):
                if predictions[ia] and target_test_seq[ia][0] in predictions[ia]:
                    supra += 1
            self.length_list.append(num)
            self.supra_list.append(supra / len(predictions) * 100)
        return self.length_list, self.supra_list

    def draw_num_of_seq(self):
        fig = plt.figure()
        plt.plot(self.length_list, self.supra_list)
        plt.ylabel('Метрика')
        plt.xlabel('Длина')
        fig.savefig('num_of_seq.png')

    def get_results_length_of_seq(self):
        self.length_tail = []
        self.supra_list = []
        target = CPTMakeData(file='data/1_part.txt', num_of_seq=50000)
        target_test_seq, test_seq, train_seq = target.make_sequences()
        model = CPT()
        model.train(train_seq, test_seq, merge=False)
        for tail in tqdm(np.arange(0.1, 1.0, 0.1)):
            predictions = model.predict(train_seq, test_seq, 0.2, 5)
            supra = 0
            print(len(target_test_seq))
            for ia in range(len(target_test_seq)):
                if predictions[ia] and target_test_seq[ia][0] in predictions[ia]:
                    supra += 1
            self.length_tail.append(tail)
            self.supra_list.append(supra / len(predictions) * 100)
        return self.length_tail, self.supra_list

    def draw_length_of_seq(self):
        fig = plt.figure()
        plt.plot(self.length_tail, self.supra_list)
        plt.ylabel('Метрика')
        plt.xlabel('Длина последоветельностей равна')
        fig.savefig('len_of_seq_fixed.png')


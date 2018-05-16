import matplotlib.pyplot as plt
from tqdm import tqdm

from algos.cpt import CPT
from implemnts.cpt_original import CPTOriginalWork


class CPTPictures:
    def __init__(self):
        pass

    def get_results_num_of_seq(self):
        self.length_list = []
        self.supra_list = []
        for length in tqdm(range(5000, 100000, 10000)):
            target = CPTOriginalWork('data/1_part.txt', 15, length)
            target_test_seq, test_seq, train_seq = target.make_sequences()
            model = CPT()
            model.train(test_seq)
            predictions, nons = model.predict(train_seq, test_seq, 5, 5)
            supra = 0
            print(len(target_test_seq))
            for ia in range(len(target_test_seq)):
                if predictions[ia] and target_test_seq[ia][0] in predictions[ia]:
                    supra += 1
            self.length_list.append(length)
            self.supra_list.append(supra/len(predictions)*100)
        return self.length_list, self.supra_list

    def draw_num_of_seq(self):
        fig = plt.figure()
        plt.plot(self.length_list, self.supra_list)
        plt.ylabel('Метрика')
        plt.xlabel('Длина')
        fig.savefig('to.png')

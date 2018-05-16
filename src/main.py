from algos.cpt import CPT
from analytics.tail_len import do_analytics
from build_data.encode_data import encode_data_whole
from directory_test.everything import CPTE
from implemnts.cpt_for_fixed_length import CPTWorkFixed
from implemnts.cpt_original import CPTOriginalWork
import numpy as np

# CPTOriginalWork('data/1_part.txt', 15)
with open('different_analytics.txt', 'w') as f:
    target = CPTOriginalWork('data/1_part.txt', 7, 50000)
    target_test_seq, test_seq, train_seq = target.predict()
    model = CPT()
    print("init")
    # train, test = model.load_files("data/train.csv", "data/test.csv")
    print('read_csv')
    model.train(test_seq)
    print('train')

    predictions, nons = model.predict(train_seq, test_seq, 5, 3)
    supra = 0
    for ia in range(len(target_test_seq)):
        if predictions[ia] and target_test_seq[ia][0] in predictions[ia]:
            supra += 1
    f.write('length% ' + str(len(target_test_seq)) + ' supra: ' + str(supra) + 'procent ' + str(
        len([i for i in predictions if i]) / len(predictions)) + '\n')
    print(('length% ' + str(len(target_test_seq)) + ' supra: ' + str(supra) + 'procent ' + str(
        len([i for i in predictions if i]) / len(predictions)) + '\n'))

d = 1

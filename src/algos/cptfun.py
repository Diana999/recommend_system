from collections import Counter

from tqdm import tqdm
import numpy as np
from algos.cpt import CPT
from implemnts.cpt_original import CPTMakeData


class CPTFun(CPT):
    def __init__(self):
        super().__init__()
        self.items_genre = {}
        sequences = CPTMakeData().sequences
        self.sequnce_genre = {}
        with open("/Users/dianagajnutdinova/PycharmProjects/untitled3/src/data/item_genre.txt", 'r') as f:
            for i in tqdm(f.readlines()):
                self.items_genre[i.split()[0]] = i.split()[1]
        for num, seq in tqdm(enumerate(sequences)):
            genres_list = []
            for item in seq:
                if self.items_genre.get(item):
                    genres_list.append(self.items_genre[item])
            self.sequnce_genre[num] = [i[0] for i in Counter(genres_list).most_common(1)]

    def score(self, counttable, key, number_of_similar_sequences, number_items_counttable, num=None, sims=None,
              elem_count=None):
        sims = 1 / len([1 for i in sims if key in i])
        score = 0
        if self.items_genre.get(key):
            if self.items_genre.get(key) in [i[0] for i in
                                             Counter([self.items_genre.get(i, 0) for i in num]).most_common(1)]:
                score += 2
            else:
                score += 1
        else:
            score += 1
        score += sims + len(elem_count)

        if counttable.get(key) is None:
            counttable[key] = score
        else:
            counttable[key] = score + counttable.get(key)
        return counttable


target = CPTMakeData()
target_test_seq, test_seq, train_seq = target.make_sequences()
model = CPTFun()
model.train(train_seq, test_seq, merge=True)
for i in np.arange(0.1, 1, 0.1):
    predictions, ttl = model.predict(train_seq, test_seq, i, 10)
    supra = 0
    print(len(target_test_seq))
    for ia in range(len(target_test_seq)):
        if predictions[ia] and target_test_seq[ia] in predictions[ia]:
            supra += 1
    with open('shows_txt', 'a') as f:
        f.write('\n')
        f.write(str(supra / len(predictions)) + ' ' + str(supra) + str(i))

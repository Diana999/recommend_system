from algos.cpt import CPT
from build_data.genre import get_items_genre_dict, get_sequence_most_n_genre
from implemnts.cpt_original import CPTMakeData


class CPTFun(CPT):
    def __init__(self):
        super().__init__()
        self.items_genre = get_items_genre_dict()
        print("got")
        self.sequnce_genre = get_sequence_most_n_genre(3)
        print("got")

    def score(self, counttable, key, number_of_similar_sequences, number_items_counttable):

        weight_level = 1 / number_of_similar_sequences
        weight_distance = 1 / number_items_counttable
        score = 1 + weight_level * 5.5 + weight_distance * 1.5
        if self.items_genre[key] in self.sequnce_genre:
            score *= 1.5
        if counttable.get(key) is None:
            counttable[key] = score
        else:
            counttable[key] = score * counttable.get(key)
        return counttable


target = CPTMakeData(file='data/1_part.txt', num_of_seq=10000)
target_test_seq, test_seq, train_seq = target.make_sequences()
model = CPT()
model.train(train_seq, test_seq, merge=False)
predictions = model.predict(train_seq, test_seq, 0.2, 2)
supra = 0
print(len(target_test_seq))
for ia in range(len(target_test_seq)):
    if predictions[ia] and target_test_seq[ia][0] in predictions[ia]:
        supra += 1
with open('shows_txt', 'a') as f:
    f.write(str(supra / len(predictions)))

model = CPTFun()
model.train(train_seq, test_seq, merge=False)
predictions = model.predict(train_seq, test_seq, 0.2, 5)
supra = 0
print(len(target_test_seq))
for ia in range(len(target_test_seq)):
    if predictions[ia] and target_test_seq[ia][0] in predictions[ia]:
        supra += 1
with open('shows_txt', 'a') as f:
    f.write('\n')
    f.write(str(supra / len(predictions)))

from algos.cpt import CPT
from build_data.build_data import CPTMakeData

for i in range(10000, 100000, 10000):
    target = CPTMakeData(num_of_seq=i)
    target_test_seq, test_seq, train_seq = target.make_sequences()
    model = CPT()
    model.train(train_seq, test_seq, merge=True)
    predictions, ttl = model.predict(train_seq, test_seq, 0.1, 10)
    supra = 0
    print(len(target_test_seq))
    for ia in range(len(target_test_seq)):
        if predictions[ia] and target_test_seq[ia] in predictions[ia]:
            supra += 1
    with open('shows.txt', 'a') as f:
        f.write('\n')
        f.write(str(supra / len(predictions)) + ' ' + str(supra) +' '+ str(i))
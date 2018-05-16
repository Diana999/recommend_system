from analytics.tail_len import do_analytics
from build_data.encode_data import encode_data_whole
from directory_test.everything import CPTE
from implemnts.cpt_for_fixed_length import CPTWorkFixed
from implemnts.cpt_original import CPTOriginalWork

# CPTOriginalWork('data/1_part.txt', 15)
target = CPTOriginalWork('data/1_part.txt')
target = target.predict()
model = CPTE()
print("init")
train, test = model.load_files("data/train.csv", "data/test.csv", merge=True)
print('read_csv')
model.train(train)
print('train')
predictions, nons, pops = model.predict(train, test, 5, 3)
supra = 0
for i in range(len(target)):
    if predictions[i] and target[i][0] in predictions[i]:
        supra += 1

мd = 1
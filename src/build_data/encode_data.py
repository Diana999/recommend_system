import itertools

from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def encode_data():
    sequences = []
    with open('data/normal_whole_data.txt', 'r') as f:
        print("Reading...")
        for line in tqdm(f.readlines()):
            sequences.append(line.split())
    print("Encoding...")
    encoder = {}
    for num, i in tqdm(enumerate(list(set(itertools.chain(*sequences))))):
        encoder[i] = num
    print("Learn how to encode...")
    with open("encodes.txt", 'w') as f:
        for i, j in encoder.items():
            f.write(str(i) + ' : ' + str(j) + '\n')

    sequences = [[encoder[i] for i in j] for j in tqdm(sequences)]
    with open('encoded_data_whole_normal.txt', 'w') as f:
        f.writelines([' '.join(list(map(str, i))) + '\n' for i in tqdm(sequences)])

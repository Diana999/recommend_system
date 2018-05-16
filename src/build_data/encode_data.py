import itertools

from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def encode_data_whole():
    sequences = []
    with open('data/data_with_len_more_2.txt', 'r') as f:
        print("Reading...")
        for line in tqdm(f.readlines()):
            sequences.append(line.split())
    print("Encoding...")
    label = LabelEncoder()
    unique_books = list(set(itertools.chain(*sequences)))
    print(len(unique_books))
    label.fit(unique_books)
    print("Learn how to encode...")
    sequences = [list(label.transform(j)) for j in tqdm(sequences)]
    with open('encoded_data_len_more_2.txt', 'w') as f:
        f.writelines([' '.join(list(map(str, i))) + '\n' for i in tqdm(sequences)])

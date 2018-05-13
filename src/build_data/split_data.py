import os


def split_file_into_n_files(file, n):
    for i in range(1,100):
        try:
            os.remove("data/{}_part.txt".format(i))
        except:
            pass
    mapping = {i: 'data/{}_part.txt'.format(i) for i in range(1, n + 1)}
    c = 1
    with open(file, 'r') as f:
        for line in f.readlines():
            with open(mapping[c], 'a') as file:
                file.write(line)
            c += 1
            if c > n:
                c = 1


def split_data_to_train_test(array, proportion):
    return array[0:int(len(array)*proportion)], array[int(len(array)*proportion):]


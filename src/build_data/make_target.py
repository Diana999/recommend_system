def split_file_into_test_and_target(file):
    with open(file, 'r') as f:
        with open('without_target.txt', 'w') as without_target:
            with open('target.txt', 'w') as target:
                for line in f.readlines():
                    line = line.split()
                    if len(line) < 2:
                        continue
                    else:
                        without_target.write(' '.join(line[:-1]))
                        target.write(' '.join(line[-1:]))


def split_list_of_seq_into_test_and_target(list_of_seq):
    return [seq[:-1] for seq in list_of_seq], [seq[-1:] for seq in list_of_seq]

from tqdm import tqdm


def delete_short_sequences():
    with open('data/whole_data.txt', 'r') as f:
        with open('data/data_with_len_more_2.txt', 'w') as d:
            for line in tqdm(f.readlines()):
                line_split = line.split()
                if len(line_split) > 2:
                    d.write(line)


delete_short_sequences()

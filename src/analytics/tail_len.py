from tqdm import tqdm

from implemnts.cpt_for_fixed_length import CPTWorkFixed
from implemnts.cpt_original import CPTOriginalWork


def do_analytics():
    smth = CPTOriginalWork('data/data_with_len_more_2.txt', 20)
    with open('data/tail_len_analytics.txt', 'w') as f:
        for tail in tqdm(range(5,6)):
            smth.predict(tail)
            #f.write(str(tail) + ' ' + str(7) + ' ' + str(smth.predict(tail)) + '\n')

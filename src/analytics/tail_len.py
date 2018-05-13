from implemnts.cpt_for_fixed_length import CPTWorkFixed


def do_analytics():
    smth = CPTWorkFixed('data/1_part.txt',15)
    with open('data/tail_len_analytics.txt', 'w') as f:
        for tail in range(2, 11):
            t = smth.predict(tail)
            f.write(str(tail) + ' ' + str(15) + ' ' + str(t) + '\n')

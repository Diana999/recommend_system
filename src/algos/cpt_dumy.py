import itertools
import math
import random
from collections import Counter

from tqdm import tqdm

from algos.cpt import CPT


class CPTDummy(CPT):
    def predict(self, data, target, k, n):
        predictions = []
        ttl = []
        for each_target in tqdm(target):
            ttl.append(each_target)
            each_target = each_target[math.ceil(len(each_target)*k):]
            tries = []
            for element in each_target:
                if self.II.get(element) is None:
                    continue
                tries += list(self.II.get(element))
            similar_sequences = []
            if len(tries) > 1:
                intersection = [i[0] for i in Counter(tries).most_common(int(len(tries)))]
            else:
                intersection = tries

            for element in intersection:
                currentnode = self.LT.get(element)
                tmp = []
                while currentnode.Item is not None:
                    tmp.append(currentnode.Item)
                    currentnode = currentnode.Parent
                similar_sequences.append(tmp)

            similar_sequences = list(set(itertools.chain(*similar_sequences)))

            predictions.append([random.choice(similar_sequences) for _ in range(n)])

        return predictions, ttl
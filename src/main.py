from websocket._abnf import numpy as np

from algos.cpt import CPT
from algos.cptfun import CPTFun
from algos.neurals import BuildNeuralLSTM

# with open("show.txt", 'a') as f:
#     for i in range(1000, 7000, 1000):
#         u = BuildNeuralLSTM(length_of_seq_fixed=8, num_of_seq=i)
#         u.make_data()
#         u.prepare_model()
#         u.fit_model()
#         u.gain_results()
# #         f.write(str(u.gain_results()) + ' ' + str(i) + '\n')
# from implemnts.cpt_normal_implement import GetCptResult
#
from implemnts.cpt_normal_implement import GetCptResult

GetCptResult('improved')

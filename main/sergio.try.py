

import numpy as np
import pandas as pd
from SERGIO.SERGIO import sergio

from INVASE import PVS
number_sc = 500

sim = sergio(number_genes=100,
             number_bins=9,
             number_sc=number_sc,
             noise_params=1,
             decays=0.8,
             sampling_state=15,
             noise_type='dpd')
sim.build_graph(input_file_taregts ='SERGIO/demo/steady-state_input_GRN.txt', input_file_regs='SERGIO/demo/steady-state_input_MRs.txt', shared_coop_state=2)
sim.simulate()
expr = sim.getExpressions()
expr.shape
expr_train = [each_expr[:, :400] for each_expr in expr]
expr_test = [each_expr[:, 400:] for each_expr in expr]
expr_clean_train = np.concatenate(expr_train, axis = 1) # #genes * #cells
expr_clean_test = np.concatenate(expr_test, axis = 1) # #genes * #cells


# sim.graph_[1]

expr.shape


import scanpy as sc

X_train = expr_clean_train.T

sc.pp.scale(X_train)
X_train.shape
# 53.0,5.0,1.0,14.0,67.0,74.0,62.0,1.7310389942765454,2.641368643483525,2.419502251829988,-3.2654552986344143,-3.0023322853469634,2.0,2.0,2.0,2.0,2.0
# gene 41.0,2.0,1.0,14.0,4.408414181539802,1.1969408910886905,2.0,2.0

Y_temp = X_train[:, 53]
Y_scale = np.interp(Y_temp, (Y_temp.min(), Y_temp.max()), (0, +1)).reshape(-1, 1)
Y_train = np.concatenate((Y_scale, 1 - Y_scale), 1)


PVS_Alg = PVS(X_train, 'Syn1', 2)

# 2. Algorithm training
PVS_Alg.train(X_train, Y_train)
Generator = PVS_Alg.generator

X_test = expr_clean_test.T
sc.pp.scale(X_test)
X_test.shape


Sel_Prob_Test = np.asarray(Generator.predict(X_test))
score = np.asarray(1. * (Sel_Prob_Test > 0.6))
expr[1,:,:]


sim_test = sergio(number_genes=100,
             number_bins=9,
             number_sc=300,
             noise_params=1,
             decays=0.8,
             sampling_state=15,
             noise_type='dpd')
sim_test.build_graph(input_file_taregts ='SERGIO/demo/steady-state_input_GRN.txt', input_file_regs='SERGIO/demo/steady-state_input_MRs.txt', shared_coop_state=2)
sim_test.simulate()
expr_test = sim_test.getExpressions()
expr_clean_ss = np.concatenate(expr, axis = 1) # #genes * #cells

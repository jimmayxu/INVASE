%load_ext autoreload
%autoreload 2

import scanpy as sc
# import anndata as ad
import numpy as np
# import pandas as pd

from INVASE import PVS, performance_metric


# Data loading
DATAFILE = '../data/thymus/'
save_path = DATAFILE+"A42.v01.yadult_raw.h5ad"
adataset = sc.read_h5ad(save_path)


# label the transcription factor in the set of all genes
# TF_list = ['TNMD','FGR','CFH']
TF_list_total = np.loadtxt('main/TF_names.txt', dtype='str').tolist()

# TF_list = np.random.choice(TF_list_total, 10)

print ("%d out of %d genes are selected highly variable genes" %(adataset.var.GeneName.size, adataset.raw.var.GeneName.size))


# TF_genes = adataset.raw.var.GeneName.isin(TF_list)

TF_genes = np.where(adataset.var.GeneName.isin(TF_list_total))[0]
print ("%d out of %d selected highly variable genes are Transcription Factors" %(TF_genes.size, adataset.var.GeneName.size))



class KeyTF:
    def __init__(self, adataset, selected_TF):
        self.selected_TF = selected_TF
        self.adataset = adataset

    def cell_category(self):
        print(self.adataset.obs['anno_final_print'].value_counts())

    def implement_invase(self, selected_gene, selected_celltype = None):
        if (selected_celltype is None):
            cells_train = np.random.choice(range(self.adataset.shape[0]) ,10000)
        else:
            cells_train = np.where(self.adataset.obs['anno_final_print'].isin(selected_celltype))[0]

        adata = self.adataset[:, selected_TF]
        adata = adata[cells_train, :]

        print('Gene %s is chosen to be the labels' % (selected_gene))
        Y_train = self.adataset.X[np.ix_(cells_train, self.adataset.var.GeneName.isin(selected_gene))]
        X_train = adata.X.toarray()

        # 1. PVS Class call
        PVS_Alg = PVS(X_train, 'Syn3', 2)

        # 2. Algorithm training
        PVS_Alg.train(X_train, Y_train)

        return(PVS_Alg)





'''

import keras
import tensorflow as tf

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)

'''

# TF_ask = np.array([ 998, 1657,  107,  773,  346,  324, 1830, 1632, 1338, 1735])

np.random.seed(0)

# X = adataset.raw.X.A[:, TF_genes]
selected_TF = np.random.choice(TF_genes,50)
# selected_TF = np.concatenate((selected_TF,TF_ask))
gene_names = adataset.var.GeneName[selected_TF].index
print(gene_names)

adataset.obs['anno_final_print'].value_counts()
# selected_celltype = ['DP(Q)', 'Treg', 'NK']
selected_celltype = ['CD4+T']

try1 = KeyTF(adataset = adataset, selected_TF = selected_TF)
PVS_Alg = try1.implement_invase(selected_gene = ['FGR','CFH'], selected_celltype = selected_celltype)




# X_test generation
celltype_test = ['CD4+T']

cells_test = np.concatenate((np.random.choice(np.where(adataset.obs['anno_final_print'].isin(celltype_test))[0],500), \
np.random.choice(range(adataset.shape[0]) ,4000) ))
# cells_test = np.where(np.random.choice(range(adataset.shape[0]) ,4000))[0]

# cells_test = np.where(adataset.obs['anno_final_print'].isin(celltype_test))[0]

adata_test = adataset[:, selected_TF]
adata_test = adata_test[cells_test, :]
X_test = adata_test.X.toarray()

#xx = adataset.obs['anno_final_print'][cells_test]


# 3. Get the selection probability on the testing set
Sel_Prob_Test = PVS_Alg.output(X_test)

import pandas as pd
Prob_Test = pd.DataFrame(Sel_Prob_Test)
Prob_Test.columns = gene_names

# Prob_Test.mean().std()

# 4. Selected features
score = 1. * (Prob_Test > 0.5)

adataset.obs['anno_final_print'][cells_test]
Prob_Test.to_csv(r'results/Prob_%s.csv'%(selected_celltype[0]))

'''
Prob_Test_load = pd.read_csv('results/Prob_%s.csv'%(selected_celltype[0]))
score_load = 1. * (Prob_Test > 0.5)

'''


gene_names[score.sum()/score.shape[0]>0.9].values

Unique_TFs = gene_names[sum(score[:500])/500>0.9].values


'''
common key TF features
array(['ZNF831', 'YBX3', 'SOX4', 'ZNF34', 'MAFB', 'AEBP1', 'NR4A2',
       'PBX1', 'ZBTB18', 'FOXO4', 'CASZ1', 'IRF8', 'HIC1'], dtype=object)
'''

# 5. Prediction
val_predict, dis_predict = PVS_Alg.get_prediction(X_test, score)

# %% Output

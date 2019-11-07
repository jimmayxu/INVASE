import numpy as np
import scanpy as sc
from scanpy import preprocessing
from keras.models import model_from_json
#%% Define PVS class
import pandas as pd


TF_list_total = np.loadtxt('main/TF_names.txt', dtype='str').tolist()

# Thymus dataset
DATAFILE = '../data/thymus/'
save_path = DATAFILE + "A42.v01.yadult_raw.h5ad"
save_path = DATAFILE + "HTA07.A04.v01.adata_fig1.h5ad"

adataset = sc.read_h5ad(save_path)
TF_genes = np.where(adataset.raw.var.GeneName.isin(TF_list_total))[0]
gene_names = adataset.raw.var.GeneName[TF_genes]

adataset.obs.iloc[:,-1].value_counts()

where = np.where(adataset.raw.var.GeneName.isin(adataset.var.GeneName[:10]))[0]

adataset.var.GeneName[:10]




# sc.pl.umap(adataset, color=  'louvain', legend_fontsize = 5)

target_Genes = ['CD4', 'CD8A',  'CD8B', 'CD19', 'CTLA4', 'TIGIT', 'GNG4', 'GNG8', 'CDK1']
toy_TF = ['RUNX3', 'ZBTB7B', 'CREB3L3', 'NR4A3', 'HIVEP2', 'HIVEP3', 'E2F3', 'IKZF4', 'FOXP3']
[TF in gene_names for TF in toy_TF]

celltype_tests = ['CD4+T','CD4+Tmem','CD8+T','CD8+Tmem']
celltype_tests = [['CD4+T','CD4+Tmem','CD8+T','CD8+Tmem'],['DP(P)','DP(Q)','DN(Q)'],['Treg'], ['B_memory', 'B_naive', 'B_plasma']]
# loadnames = ['|'.join(celltype_tests) + '@' + x for x in selected_gene]

key_TF = pd.DataFrame(columns=['target_gene', 'testing_cells', 'key TFs'])
foldername = 'raw_counts'

for target_gene in target_Genes:
    # load json and create model
    # load_name = '|'.join(cells_test) + '@' + target_gene
    load_name = 'all_cells' + '@' + target_gene
    json_file = open('results/%s/generator_%s.json' % (foldername,load_name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    Generator = model_from_json(loaded_model_json)
    # load weights into new model
    Generator.load_weights('results/%s/weights_%s.h5' % (foldername,load_name))
    print("Loaded model from disk")


    # 3. Get the selection probability on the testing set
    # X_test generationSin

    for celltype_test in celltype_tests:
        #cells_test = np.concatenate((np.random.choice(np.where(adataset.obs['anno_final_print'].isin(celltype_test))[0], 500),
        #np.random.choice(range(adataset.shape[0]), 4000)))

        cells_test = np.where(adataset.obs.iloc[:,-1].isin(celltype_test))[0]

        X_test = adataset.raw.X[np.ix_(cells_test, TF_genes)]

        Sel_Prob_Test = np.asarray(Generator.predict(X_test))

        Prob_Test = pd.DataFrame(Sel_Prob_Test)
        Prob_Test.columns = gene_names
        score = np.asarray(1. * (Prob_Test > 0.6))

        key_TF = key_TF.append({'target_gene': target_gene,
                       'testing_cells': celltype_test,
                       'key TFs': np.asarray(gene_names[score.sum(0)/len(score)>0.9].values)}, ignore_index=True)
        print('end')



np.transpose(pd.DataFrame(np.unique(np.concatenate(key_TF['TF'][key_TF['target_gene']=='FGR'].tolist()), return_counts=True)))




from sklearn.metrics import mean_squared_error, mutual_info_score
mean_squared_error(Generator.get_weights()[5], Generator2.get_weights()[5])
mutual_info_score(Generator.get_weights()[5], Generator2.get_weights()[5])
mutual_info_score(Generator.get_weights()[1], Generator2.get_weights()[1])


['RUNX3' in listt for listt in key_TF['key TFs'][key_TF['target_gene'] == 'CD8A']]



'''
import pandas as pd
Prob_Test_CD4 = pd.read_csv('results/Prob_CD4+T_ALL.csv')
score_load_CD4 = 1. * (Prob_Test_CD4 > 0.8)

Prob_Test_DPQ = pd.read_csv('results/Prob_DP(Q).csv')
score_load_DPQ = 1. * (Prob_Test_DPQ > 0.8)


gene_names[score_load[:500].sum()/500>0.9].values
'''

# 5. Prediction
# val_predict, dis_predict = Generator.get_prediction(X_test, score)

# %% Output


## Plot and visualisation
import matplotlib.pyplot as plt
for gene in target_Genes:
    fig = plt.figure()
    fig.suptitle(gene, fontsize=16)
    for i, cells in enumerate([['CD4+T','CD4+Tmem','CD8+T','CD8+Tmem'],['DP(P)','DP(Q)','DN(Q)'],['Treg']]):
        x = adataset.raw.X[np.ix_(adataset.obs['anno_final_print'].isin(cells), adataset.raw.var.GeneName == gene)]
        x = x.toarray()
        xx = adataset.X[np.ix_(adataset.obs['anno_final_print'].isin(cells), adataset.var.GeneName == gene)]
        num_bins = 10
        ax = plt.subplot(3, 2, 2*(i+1)-1)
        ax.set_title('raw counts')
        #plt.figtext(0.5, 0.5*i+0.1, cells, ha="center",
        #           va="top", fontsize=14, color="r")

        n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.5)
        ax = plt.subplot(3, 2, 2*(i+1))
        ax.set_title('normalised')
        n, bins, patches = plt.hist(xx, num_bins, facecolor='blue', alpha=0.5)
    plt.show()
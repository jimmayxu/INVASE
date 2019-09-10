import numpy as np
import scanpy as sc
from keras.models import model_from_json
import pandas as pd
#%% Define PVS class


TF_list_total = np.loadtxt('main/TF_names.txt', dtype='str').tolist()

# Thymus dataset
DATAFILE = '../data/thymus/'
save_path = DATAFILE + "A42.v01.yadult_raw.h5ad"

adataset = sc.read_h5ad(save_path)
TF_genes = np.where(adataset.var.GeneName.isin(TF_list_total))[0]
gene_names = adataset.var.GeneName[TF_genes]

adataset.obs['anno_final_print'].value_counts()
###

# T cell dataset
DATAFILE = '../data/'
save_path = DATAFILE + "T-cell.h5ad"

adataset = sc.read_h5ad(save_path)
TF_genes = np.where(adataset.var.GeneName.isin(TF_list_total))[0]
gene_names = adataset.var.GeneName[TF_genes]
print(gene_names.size)
adataset.obs['anno_final_print'].value_counts()

###



loadnames = [''.join(['CD4+T|CD4+Tmem|CD8+T|CD8+Tmem|B_mature@',x]) for x in ['CD8A','CD8B','FGR']]

key_TF = pd.DataFrame(columns=['name', 'TF'])

for i, load_name in enumerate(loadnames):


    # load json and create model
    json_file = open('results/generator_%s.json' % load_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    Generator = model_from_json(loaded_model_json)
    # load weights into new model
    Generator.load_weights('results/weights_%s.h5' % load_name)
    print("Loaded model from disk")


    # 3. Get the selection probability on the testing set
    # X_test generation
    celltype_test = ['CD4+T']


    cells_test = np.concatenate((np.random.choice(np.where(adataset.obs['anno_final_print'].isin(celltype_test))[0], 500),
    np.random.choice(range(adataset.shape[0]), 4000)))

    cells_test = np.where(adataset.obs['anno_final_print'].isin(celltype_test))[0]

    X_test = adataset.X[np.ix_(cells_test, TF_genes)]


    Sel_Prob_Test = np.asarray(Generator.predict(X_test))

    Prob_Test = pd.DataFrame(Sel_Prob_Test)
    Prob_Test.columns = gene_names
    score = np.asarray(1. * (Prob_Test > 0.8))



    key_TF.loc[i] = [loadnames , np.asarray(gene_names[score.sum(0)/len(score)>0.9].values)]


    '''
    import pandas as pd
    Prob_Test_CD4 = pd.read_csv('results/Prob_CD4+T_ALL.csv')
    score_load_CD4 = 1. * (Prob_Test_CD4 > 0.8)
    
    Prob_Test_DPQ = pd.read_csv('results/Prob_DP(Q).csv')
    score_load_DPQ = 1. * (Prob_Test_DPQ > 0.8)
    
    
    gene_names[score_load[:500].sum()/500>0.9].values
    '''






    '''
    common key TF features
    array(['ZNF831', 'YBX3', 'SOX4', 'ZNF34', 'MAFB', 'AEBP1', 'NR4A2',
           'PBX1', 'ZBTB18', 'FOXO4', 'CASZ1', 'IRF8', 'HIC1'], dtype=object)
    '''

    # 5. Prediction
    # val_predict, dis_predict = PVS_Alg.get_prediction(X_test, score)

    # %% Output

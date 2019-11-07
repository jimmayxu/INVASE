#!/usr/bin/env python:

# %% Necessary packages
# GPU node choice
import sys, ast, os
import time
import scanpy as sc
import numpy as np

sys.path.append(os.getcwd())
from INVASE import KeyTF

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

'''
pipenv shell
cd PycharmProject/INVASE/

python3 main/run_file.py "['CD4', 'CD8A',  'CD8B', 'CD19', 'CTLA4', 'TIGIT', 'GNG4', 'GNG8', 'CDK1']" 2> log/log18092019.txt
'''

#exec(open('INVASE.py').read())
# selected_gene check:
# 'ZBTB16' in TF_list_total

if __name__ == '__main__':
    arg = ast.literal_eval(sys.argv[1])
    target_Genes = arg
    selected_celltype = None
    # selected_celltype = arg[1] #['CD4+T', 'CD4+Tmem', 'CD8+T', 'CD8+Tmem', 'B_mature']

    assert isinstance(target_Genes, list)
    # assert isinstance(selected_celltype, list)

    print('target gene selected: %s' % (target_Genes))
    # print('cell type selected: %s' % (selected_celltype))
    # %% Data loading
    print("data loading")
    DATAFILE = '../data/thymus/'
    # save_path = DATAFILE + "A42.v01.yadult_raw.h5ad"
    save_path = DATAFILE + "HTA07.A04.v01.adata_fig1.h5ad"

    adataset = sc.read_h5ad(save_path)
    print ('data is loaded')

    assert sum(adataset.raw.var.GeneName.isin(target_Genes)) == len(target_Genes)

    TF_list_total = np.loadtxt('main/TF_names.txt', dtype='str').tolist()

    #print("%d out of %d genes are selected highly variable genes" % (
    # adataset.var.GeneName.size, adataset.raw.var.GeneName.size))



    # %% run INVASE model
    try1 = KeyTF(adataset = adataset, target_Genes = target_Genes, TF_list_total = TF_list_total, raw_counts = True)
    try1.filter_matrix()
    for gene in range(len(target_Genes)):
        t0 = time.time()
        PVS_Alg = try1.implement_invase(gene=gene)

        t = (time.time() - t0)/60
        save_name = 'all_cells' + '@' + target_Genes[gene]
        #save_name = '|'.join(selected_celltype) + '@' + gene

        model = PVS_Alg.generator
        model.name = dict(selected_celltype = selected_celltype, target_Gene = gene, time_spend = t)


        model_json = model.to_json()
        with open("results/generator_%s.json" % save_name, "w") as json_file:
            json_file.write(model_json)
        model.save_weights("results/weights_%s.h5" % save_name)
        print("Saved model: %s" % save_name)
        print('cell type selected: %s' % ('all cells'))
        print('target gene selected: %s' % (target_Genes[gene]))
        print('Time spent: %.2f minutes' % t)
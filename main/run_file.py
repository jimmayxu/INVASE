#!/usr/bin/python

# %% Necessary packages
# GPU node choice
import sys, ast, os
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

'''
pipenv shell
cd PycharmProject/INVASE/
python main/run_file.py "[['CD8A',  'CD8B', 'FGR'], ['CD4+T', 'CD4+Tmem', 'CD8+T', 'CD8+Tmem', 'B_mature']]"

python main/run_file.py "[['CD8A',  'CD8B', 'FGR', 'ABCA1', 'A2M']]" > log/log10092019.txt
'''


import numpy as np
import scanpy as sc

exec(open('INVASE.py').read())

# selected_gene = ['CD8A',  'CD8B', 'FGR', 'ABCA1', 'A2M']

if __name__ == '__main__':
    arg = ast.literal_eval(sys.argv[1])
    selected_gene = arg[0] #['CD8A', ]
    selected_celltype = None
    # selected_celltype = arg[1] #['CD4+T', 'CD4+Tmem', 'CD8+T', 'CD8+Tmem', 'B_mature']

    assert isinstance(selected_gene, list)
    # assert isinstance(selected_celltype, list)

    print('target gene selected: %s' % (selected_gene))
    # print('cell type selected: %s' % (selected_celltype))
    # %% Data loading
    DATAFILE = '../data/thymus/'
    save_path = DATAFILE + "A42.v01.yadult_raw.h5ad"
    adataset = sc.read_h5ad(save_path)


    assert sum(adataset.var.GeneName.isin(selected_gene)) == len(selected_gene)

    TF_list_total = np.loadtxt('main/TF_names.txt', dtype='str').tolist()

    print("%d out of %d genes are selected highly variable genes" % (
    adataset.var.GeneName.size, adataset.raw.var.GeneName.size))

    TF_genes = np.where(adataset.var.GeneName.isin(TF_list_total))[0]
    print("%d out of %d selected highly variable genes are Transcription Factors" % (
    TF_genes.size, adataset.var.GeneName.size))

    # %% run INVASE model
    try1 = KeyTF(adataset = adataset, selected_TF = TF_genes)

    for i in range(len(selected_gene)):
        t0 = time.time()
        PVS_Alg = try1.implement_invase(selected_gene=selected_gene[i], selected_celltype=selected_celltype)

        t = (time.time() - t0)/60
        save_name = 'all_cells' + '@' + selected_gene[i]
        #save_name = '|'.join(selected_celltype) + '@' + selected_gene[i]

        model = PVS_Alg.generator
        model.name = dict(selected_celltype = selected_celltype, selected_gene = selected_gene, time_spend = t)


        model_json = model.to_json()
        with open("results/generator_%s.json" % save_name, "w") as json_file:
            json_file.write(model_json)
        model.save_weights("results/weights_%s.h5" % save_name)
        print("Saved model: %s" % save_name)
        print('cell type selected: %s' % (selected_celltype))
        print('target gene selected: %s' % (selected_gene[i]))
        print('Time spent: %.2f minutes' % t)
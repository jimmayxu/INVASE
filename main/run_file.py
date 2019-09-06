# %% Necessary packages
# GPU node choice
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "6"


import numpy as np
import scanpy as sc

exec(open('INVASE.py').read())
# importlib.import_module('INVASE')
# from INVASE import KeyTF


if __name__ == '__main__':
    selected_celltype = ['CD4+T']

    print('cell type selected: %s' % (selected_celltype))
    # %% Data loading
    DATAFILE = '../data/thymus/'
    save_path = DATAFILE + "A42.v01.yadult_raw.h5ad"
    adataset = sc.read_h5ad(save_path)

    TF_list_total = np.loadtxt('main/TF_names.txt', dtype='str').tolist()

    print("%d out of %d genes are selected highly variable genes" % (
    adataset.var.GeneName.size, adataset.raw.var.GeneName.size))

    TF_genes = np.where(adataset.var.GeneName.isin(TF_list_total))[0]
    print("%d out of %d selected highly variable genes are Transcription Factors" % (
    TF_genes.size, adataset.var.GeneName.size))

    # %% run INVASE model
    try1 = KeyTF(adataset = adataset, selected_TF = TF_genes)

    PVS_Alg = try1.implement_invase(selected_gene=['FGR', 'CFH'], selected_celltype=selected_celltype)

    save_name = selected_celltype[0]

    model = PVS_Alg.generator
    model_json = model.to_json()
    with open("results/generator_%s.json" % save_name, "w") as json_file:
        json_file.write(model_json)
    model.save_weights("results/weights_%s.h5" % save_name)
    print("Saved model to disk")
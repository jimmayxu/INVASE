#!/usr/bin/python
from keras.layers import Input, Dense, Multiply
from keras.layers import BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import regularizers
from keras import backend as K

import scanpy as sc
def main():
    print('dadfads')

    DATAFILE = '../data/thymus/'
    save_path = DATAFILE + "A42.v01.yadult_raw.h5ad"
    print('dadfads')
    adataset = sc.read_h5ad(save_path)
    print('dadfads')


if __name__ == '__main__':
    main()
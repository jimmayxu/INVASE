import numpy as np
import sys, ast, time, os, random
import scanpy as sc
from scipy.special import softmax
sys.path.append(os.getcwd())

from INVASE import PVS


"""
python3 main/run_sergio1.py "[55, 48, 12, 53, 41, 29, 21]" "'bin2'" 2> log/log12112019.txt
target_Genes = [55, 48, 12, 53, 41, 29, 21]
"""

if __name__ == '__main__':
    target_Genes  = ast.literal_eval(sys.argv[1])
    foldername = ast.literal_eval(sys.argv[2])

    assert isinstance(target_Genes, list)
    # assert isinstance(selected_celltype, list)

    print('target gene selected: %s' % (target_Genes))

    print("loading simualated data")

    number_sc = 2500
    number_bins = 2
    number_train = 2000
    expr_temp = np.loadtxt('SERGIO/raw_2500sc.txt', dtype=float)
    expr = expr_temp.reshape(number_bins, -1, number_sc)

    expr_train = [each_expr[:, :number_train] for each_expr in expr]
    expr_test = [each_expr[:, number_train:] for each_expr in expr]
    expr_clean_train = np.concatenate(expr_train, axis=1).T  # cells * genes
    expr_clean_test = np.concatenate(expr_test, axis=1).T  # cells * genes

    print("simulated data is loaded")
    X = np.concatenate((expr_clean_train, expr_clean_test), axis=0)
    sc.pp.scale(X)
    np.savetxt('SERGIO/scale_2500sc.txt', X)
    X_train_full = X[:(number_train * number_bins), :]
    # X_train = np.delete(X_train_full, target_Genes, 1)
    X_train = X_train_full


    def sigmoid(X):
        return 1 / (1 + np.exp(-X))


    for target_Gene in target_Genes:
        """
        target_Gene = target_Genes[0]
        Y_temp = X_train[:, target_Gene]
        Y_scale = np.interp(Y_temp, (Y_temp.min(), Y_temp.max()), (0, +1)).reshape(-1, 1)
        Y_train = np.concatenate((Y_scale, Y_scale), 1)
        """
        Y_temp = X_train_full[:, target_Gene]
        Y_scale = sigmoid(Y_temp).reshape(-1, 1)
        Y_train = np.concatenate((Y_scale, 1 - Y_scale), 1)

        print("Target gene: %d" % (target_Gene))
        PVS_Alg = PVS(X_train, activation = 'selu', output_shape=2)

        # 2. Algorithm training
        t0 = time.time()
        loss = PVS_Alg.train(X_train, Y_train)
        t = (time.time() - t0) / 60



        save_name = 'sergio@%d' % (target_Gene)
        model = PVS_Alg.generator
        model.name = dict(number_genes=100,
                          number_bins=number_bins,
                          number_sc=number_sc,
                          noise_params=1,
                          decays=0.8,
                          sampling_state=15,
                          noise_type='dpd',
                          target_Gene=target_Gene, time_spend=t)

        model_json = model.to_json()
        save_dir = "sergio_results/%s" % (foldername)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        with open("%s/generator_%s.json" % (save_dir, save_name), "w") as json_file:
            json_file.write(model_json)
        model.save_weights("%s/weights_%s.h5" % (save_dir, save_name))
        loss.to_pickle("%s/loss_%s.pkl" % (save_dir, save_name))

        print("Saved model: %s" % save_name)
        print('target gene selected: %s' % (target_Gene))
        print('Time spent: %.2f minutes' % t)
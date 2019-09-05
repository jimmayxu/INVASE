# %% Main Function
#%load_ext autoreload
#%autoreload 2


if __name__ == '__main__':

    # Data generation function import
    from Data_Generation import generate_data
    from INVASE import PVS, performance_metric
    import numpy as np


    # %% Parameters
    # Synthetic data type
    idx = 4
    data_sets = ['Syn1', 'Syn2', 'Syn3', 'Syn4', 'Syn5', 'Syn6']
    data_type = data_sets[idx]

    # Data output can be either binary (Y) or Probability (Prob)
    data_out_sets = ['Y', 'Prob']

    data_out = data_out_sets[0]
    data_out = 'Prob'

    # Number of Training and Testing samples
    train_N = 10000
    test_N = 10000

    # Seeds (different seeds for training and testing)
    train_seed = 0
    test_seed = 1


    # %% Data Generation (Train/Test)
    def create_data(data_type, data_out):
        x_train, y_train, g_train = generate_data(n=train_N, data_type=data_type, seed=train_seed, out=data_out)
        x_test, y_test, g_test = generate_data(n=test_N, data_type=data_type, seed=test_seed, out=data_out)

        return x_train, y_train, g_train, x_test, y_test, g_test


    x_train, y_train, g_train, x_test, y_test, g_test = create_data(data_type, data_out)

    y_train = np.c_[y_train, np.random.randn(train_N)]

    #x_train = x_train +10
    #x_test = x_test + 10

    y_train = y_train[:,0].reshape(-1,1)

    # %%
    # 1. PVS Class call
    PVS_Alg = PVS(x_train, data_type)

    # 2. Algorithm training
    PVS_Alg.train(x_train, y_train)

    # 3. Get the selection probability on the testing set
    Sel_Prob_Test = PVS_Alg.output(x_test)

    # 4. Selected features
    score = 1. * (Sel_Prob_Test > 0.5)

    # 5. Prediction
    val_predict, dis_predict = PVS_Alg.get_prediction(x_test, score)

    # %% Output

    TPR_mean, FDR_mean, TPR_std, FDR_std = performance_metric(score, g_test)


    print('TPR mean: ' + str(np.round(TPR_mean, 1)) + '\%, ' + 'TPR std: ' + str(np.round(TPR_std, 1)) + '\%, ')
    print('FDR mean: ' + str(np.round(FDR_mean, 1)) + '\%, ' + 'FDR std: ' + str(np.round(FDR_std, 1)) + '\%, ')

    from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
    # %% Prediction Results
    Predict_Out = np.zeros([20, 3, 2])

    for i in range(20):
        # different teat seed
        test_seed = i + 2
        _, _, _, x_test, y_test, _ = create_data(data_type, data_out)

        # 1. Get the selection probability on the testing set
        Sel_Prob_Test = PVS_Alg.output(x_test)

        # 2. Selected features
        score = 1. * (Sel_Prob_Test > 0.5)

        # 3. Prediction
        val_predict, dis_predict = PVS_Alg.get_prediction(x_test, score)

        # 4. Prediction Results
        Predict_Out[i, 0, 0] = roc_auc_score(y_test[:, 1], val_predict[:, 1])
        Predict_Out[i, 1, 0] = average_precision_score(y_test[:, 1], val_predict[:, 1])
        Predict_Out[i, 2, 0] = accuracy_score(y_test[:, 1], 1. * (val_predict[:, 1] > 0.5))

        Predict_Out[i, 0, 1] = roc_auc_score(y_test[:, 1], dis_predict[:, 1])
        Predict_Out[i, 1, 1] = average_precision_score(y_test[:, 1], dis_predict[:, 1])
        Predict_Out[i, 2, 1] = accuracy_score(y_test[:, 1], 1. * (dis_predict[:, 1] > 0.5))

    # Mean / Var of 20 different testing sets
    Output = np.round(np.concatenate((np.mean(Predict_Out, 0), np.std(Predict_Out, 0)), axis=1), 4)

    print(Output)




    cost = PVS_Alg.discriminator.evaluate(x = [x_test, Sel_Prob_Test], y = y_test, batch_size = 50)
    PVS_Alg.valfunction.evaluate(x = x_test, y = y_test, batch_size = 50)
    PVS_Alg.discriminator.summary()
    PVS_Alg.valfunction.summary()
    xx = PVS_Alg.valfunction.layers[1].get_weights()



    idx = np.random.randint(0, x_train.shape[0], PVS_Alg.batch_size)
    x_batch = x_train[idx, :]
    y_batch = y_train[idx, :]
    gen_prob = PVS_Alg.generator.predict(x_batch)
    sel_prob = PVS_Alg.Sample_M(gen_prob)

    d_loss = PVS_Alg.discriminator.train_on_batch([x_batch, sel_prob], y_batch)
    # Compute the prediction of the critic based on the sampled features (used for generator training)
    dis_prob = PVS_Alg.discriminator.predict([x_batch, sel_prob])

    # %% Train Valud function

    # Compute the prediction of the critic based on the sampled features (used for generator training)
    val_prob = PVS_Alg.valfunction.predict(x_batch)

    # Train the discriminator
    v_loss = PVS_Alg.valfunction.train_on_batch(x_batch, y_batch)

    # %% Train Generator
    # Use three things as the y_true: sel_prob, dis_prob, and ground truth (y_batch)
    y_batch_final = np.concatenate((sel_prob, np.asarray(dis_prob)[:,:2], np.asarray(val_prob)[:,:2], y_batch[:,:2]), axis=1)

    g_loss = PVS_Alg.generator.train_on_batch(x_batch, y_batch_final)


if __name__ == '__main__':
    # Data generation function import
    from Data_Generation import generate_data
    from INVASE import PVS, performance_metric

    # %% Parameters
    # Synthetic data type
    idx = 0
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

    #x_train = x_train +10
    #x_test = x_test + 10

    y_train = y_train[:,0].reshape(-1,1)

    # %%
    # 1. PVS Class call
    PVS_Alg = PVS(x_train, data_type)

    # 2. Algorithm training
    PVS_Alg.train(x_train, y_train)
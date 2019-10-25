'''
Written by Jinsung Yoon
Date: Jan 1th 2019
INVASE: Instance-wise Variable Selection using Neural Networks Implementation on Synthetic Datasets
Reference: J. Yoon, J. Jordon, M. van der Schaar, "IINVASE: Instance-wise Variable Selection using Neural Networks," International Conference on Learning Representations (ICLR), 2019.
Paper Link: https://openreview.net/forum?id=BJg_roAcK7
Contact: jsyoon0823@g.ucla.edu

---------------------------------------------------

Instance-wise Variable Selection (INVASE) - with baseline networks
'''

#%% Necessary packages
# 1. Keras
from keras.layers import Input, Dense, Multiply
from keras.layers import BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import regularizers
from keras import backend as K

# 2. Others
import tensorflow as tf
import numpy as np
import scanpy as sc


#%% Define PVS class
class PVS():

    # 1. Initialization
    '''
    x_train: training samples
    data_type: Syn1 to Syn 6
    '''
    def __init__(self, x_train, data_type, output_shape):
        self.latent_dim1 = 100      # Dimension of actor (generator) network
        self.latent_dim2 = 200      # Dimension of critic (discriminator) network

        self.batch_size = 1000      # Batch size
        self.epochs = 10000         # Epoch size (large epoch is needed due to the policy gradient framework)
        self.lamda = 0.1            # Hyper-parameter for the number of selected features

        self.input_shape = x_train.shape[1]     # Input dimension

        # final layer dimension
        self.output_shape = output_shape

        # Actionvation. (For Syn1 and 2, relu, others, selu)
        self.activation = 'relu' if data_type in ['Syn1','Syn2'] else 'selu'

        # Use Adam optimizer with learning rate = 0.0001
        optimizer = Adam(0.0001)

        # Build and compile the discriminator (critic)
        self.discriminator = self.build_discriminator()
        # Use categorical cross entropy as the loss
#        self.discriminator.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
        self.discriminator.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['acc'])

        # Build the generator (actor)
        self.generator = self.build_generator()
        # Use custom loss (my loss)
        self.generator.compile(loss=self.my_loss, optimizer=optimizer)

        # Build and compile the value function
        self.valfunction = self.build_valfunction()
        # Use categorical cross entropy as the loss
#        self.valfunction.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
        self.valfunction.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['acc'])

    #%% Custom loss definition
    def my_loss(self, y_true, y_pred):

        # dimension of the features
        d = y_pred.shape[1]

        # Put all three in y_true
        # 1. selected probability
        sel_prob = y_true[:,:d]
        # 2. discriminator output
        dis_prob = y_true[:,d:(d+self.output_shape)]
        # 3. valfunction output
        val_prob = y_true[:,(d+self.output_shape):(d+self.output_shape*2)]
        # 4. ground truth
        y_final = y_true[:,(d+self.output_shape*2):]

        # A1. Compute the rewards of the actor network
        Reward1 = tf.reduce_sum(y_final * tf.log(dis_prob + 1e-8), axis = 1)

        # A2. Compute the rewards of the actor network
        Reward2 = tf.reduce_sum(y_final * tf.log(val_prob + 1e-8), axis = 1)

        # Difference is the rewards
        Reward = Reward1 - Reward2

        # B. Policy gradient loss computation.
        loss1 = Reward * tf.reduce_sum( sel_prob * K.log(y_pred + 1e-8) + (1-sel_prob) * K.log(1-y_pred + 1e-8), axis = 1) - self.lamda * tf.reduce_mean(y_pred, axis = 1)

        # C. Maximize the loss1
        loss = tf.reduce_mean(-loss1)

        return loss

    #%% Generator (Actor)
    def build_generator(self):

        model = Sequential()

        model.add(Dense(100, activation=self.activation, name = 's/dense1', kernel_regularizer=regularizers.l2(1e-3), input_dim = self.input_shape))
        model.add(Dense(100, activation=self.activation, name = 's/dense2', kernel_regularizer=regularizers.l2(1e-3)))
        model.add(Dense(self.input_shape, activation = 'sigmoid', name = 's/dense3', kernel_regularizer=regularizers.l2(1e-3)))

        model.summary()

        feature = Input(shape=(self.input_shape,), dtype='float32')
        select_prob = model(feature)

        return Model(feature, select_prob)

    #%% Discriminator (Critic)
    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(200, activation=self.activation, name = 'dense1', kernel_regularizer=regularizers.l2(1e-3), input_dim = self.input_shape))
        model.add(BatchNormalization())     # Use Batch norm for preventing overfitting
        model.add(Dense(200, activation=self.activation, name = 'dense2', kernel_regularizer=regularizers.l2(1e-3)))
        model.add(BatchNormalization())
        model.add(Dense(self.output_shape, activation ='softmax', name = 'dense3', kernel_regularizer=regularizers.l2(1e-3)))

        model.summary()

        # There are two inputs to be used in the discriminator
        # 1. Features
        feature = Input(shape=(self.input_shape,), dtype='float32')
        # 2. Selected Features
        select = Input(shape=(self.input_shape,), dtype='float32')

        # Element-wise multiplication
        model_input = Multiply()([feature, select])
        prob = model(model_input)

        return Model([feature, select], prob)

    #%% Value Function
    def build_valfunction(self):

        model = Sequential()

        model.add(Dense(200, activation=self.activation, name = 'v/dense1', kernel_regularizer=regularizers.l2(1e-3), input_dim = self.input_shape))
        model.add(BatchNormalization())     # Use Batch norm for preventing overfitting
        model.add(Dense(200, activation=self.activation, name = 'v/dense2', kernel_regularizer=regularizers.l2(1e-3)))
        model.add(BatchNormalization())
        model.add(Dense(self.output_shape, activation ='softmax', name = 'v/dense3', kernel_regularizer=regularizers.l2(1e-3)))

        model.summary()

        # There are one inputs to be used in the value function
        # 1. Features
        feature = Input(shape=(self.input_shape,), dtype='float32')

        # Element-wise multiplication
        prob = model(feature)

        return Model(feature, prob)

    #%% Sampling the features based on the output of the generator
    def Sample_M(self, gen_prob):

        # Shape of the selection probability
        n = gen_prob.shape[0]
        d = gen_prob.shape[1]

        # Sampling
        samples = np.random.binomial(1, gen_prob, (n,d))

        return samples

    #%% Training procedure
    def train(self, x_train, y_train):

        # For each epoch (actually iterations)
        for epoch in range(self.epochs):

            #%% Train Discriminator
            # Select a random batch of samples
            idx = np.random.randint(0, x_train.shape[0], self.batch_size)
            x_batch = x_train[idx,:]
            y_batch = y_train[idx,:]

            # Generate a batch of probabilities of feature selection
            gen_prob = self.generator.predict(x_batch)

            # Sampling the features based on the generated probability
            sel_prob = self.Sample_M(gen_prob)

            # Compute the prediction of the critic based on the sampled features (used for generator training)
            dis_prob = self.discriminator.predict([x_batch, sel_prob])

            # Train the discriminator
            d_loss = self.discriminator.train_on_batch([x_batch, sel_prob], y_batch)

            #%% Train Valud function

            # Compute the prediction of the critic based on the sampled features (used for generator training)
            val_prob = self.valfunction.predict(x_batch)

            # Train the discriminator
            v_loss = self.valfunction.train_on_batch(x_batch, y_batch)

            #%% Train Generator
            # Use three things as the y_true: sel_prob, dis_prob, and ground truth (y_batch)
            y_batch_final = np.concatenate( (sel_prob, np.asarray(dis_prob), np.asarray(val_prob), y_batch), axis = 1 )

            # Train the generator
            g_loss = self.generator.train_on_batch(x_batch, y_batch_final)

            #%% Plot the progress
            dialog = 'Epoch: ' + str(epoch) + ', d_loss (Acc)): ' + str(d_loss[1]) + ', v_loss (Acc): ' + str(v_loss[1]) + ', g_loss: ' + str(np.round(g_loss,4))

            if epoch % 1000 == 0:
                print(dialog)

    #%% Selected Features
    def output(self, x_train):

        gen_prob = self.generator.predict(x_train)

        return np.asarray(gen_prob)

    #%% Prediction Results
    def get_prediction(self, x_train, m_train):

        val_prediction = self.valfunction.predict(x_train)

        dis_prediction = self.discriminator.predict([x_train, m_train])

        return np.asarray(val_prediction), np.asarray(dis_prediction)



# %% Define Key TF selection class

class KeyTF():
    def __init__(self, adataset, target_Genes, TF_list_total, raw_counts = False):
        self.adataset = adataset
        self.TF_list_total = TF_list_total
        self.target_Genes = target_Genes
        self.raw_counts = raw_counts

    # def cell_category(self):
    #    print(self.adataset.obs.iloc[:,-1].value_counts())

    def filter_matrix(self):
        if self.raw_counts:
            X = self.adataset.raw.X
            GeneName = self.adataset.raw.var.GeneName
            TF_genes = np.where(GeneName.isin(self.TF_list_total))[0]
            print("%d out of total %d Transcription factors are involved in the dataset" % (
                TF_genes.size, len(self.TF_list_total)))

            Y_temp = X[:, np.where(GeneName.isin(self.target_Genes))[0]]
            self.Y_all = sc.pp.scale(Y_temp)
            X_temp = X[:, TF_genes]
            self.X_train = sc.pp.scale(X_temp)
        else:
            X = self.adataset.X
            GeneName = self.adataset.var.GeneName
            TF_genes = np.where(GeneName.isin(self.TF_list_total))[0]
            print("%d out of total %d Transcription factors are involved in the dataset" % (
                TF_genes.size, len(self.TF_list_total)))
            print("%d out of %d selected highly variable genes are Transcription Factors" % (
             TF_genes.size, GeneName.size))

            self.Y_all = X[:, GeneName == self.target_Genes]
            self.X_train = X[:, TF_genes]

    def implement_invase(self, gene):
        #    cells_train = np.where(self.adataset.obs['anno_final_print'].isin(selected_celltype))[0]

        print('target gene selected: %s' % (self.target_Genes[gene]))
        Y_temp = self.Y_all[:, gene]
        Y_scale = np.interp(Y_temp, (Y_temp.min(), Y_temp.max()), (0, +1)).reshape(-1,1)
        Y_train = np.concatenate((Y_scale, 1 - Y_scale), 1)
        # 1. PVS Class call
        PVS_Alg = PVS(self.X_train, 'Syn1', 2)

        # 2. Algorithm training
        PVS_Alg.train(self.X_train, Y_train)

        return (PVS_Alg)

if __name__ == '__main__':

#%% Performance Metrics
    def performance_metric(score, g_truth):

        n = len(score)
        Temp_TPR = np.zeros([n,])
        Temp_FDR = np.zeros([n,])

        for i in range(n):

            # TPR
            TPR_Nom = np.sum(score[i,:] * g_truth[i,:])
            TPR_Den = np.sum(g_truth[i,:])
            Temp_TPR[i] = 100 * float(TPR_Nom)/float(TPR_Den+1e-8)

            # FDR
            FDR_Nom = np.sum(score[i,:] * (1-g_truth[i,:]))
            FDR_Den = np.sum(score[i,:])
            Temp_FDR[i] = 100 * float(FDR_Nom)/float(FDR_Den+1e-8)

        return np.mean(Temp_TPR), np.mean(Temp_FDR), np.std(Temp_TPR), np.std(Temp_FDR)


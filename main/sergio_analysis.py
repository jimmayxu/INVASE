import numpy as np
import pandas as pd
from keras.models import model_from_json
import matplotlib.pyplot as plt
from sergio import sergio

print("Generating simulated data")


number_sc = 2500
number_bins = 2
sim = sergio(number_genes=100,
             number_bins=number_bins,
             number_sc=number_sc,
             noise_params=1,
             decays=0.8,
             sampling_state=15,
             noise_type='dpd')
sim.build_graph(input_file_taregts='SERGIO/steady-state_input_GRN.txt',
                input_file_regs='SERGIO/steady-state_input_MRs2bin.txt', shared_coop_state=2)
sim.simulate()
expr = sim.getExpressions()
expr_temp = expr.reshape(number_bins,-1)
np.savetxt('SERGIO/raw_2500sc.txt', expr_temp)

""" 
55.0,5.0,1.0,14.0,67.0,74.0,62.0,2.070749455708552,4.604694219890629,3.2268921553197423,-1.3621739312081949,-3.1224024336254876,2.0,2.0,2.0,2.0,2.0
48.0,2.0,1.0,14.0,1.6567107631183235,1.1876420058853254,2.0,2.0
12.0,2.0,1.0,14.0,3.6091331513531637,4.179067639289042,2.0,2.0
53.0,5.0,1.0,14.0,67.0,74.0,62.0,1.7310389942765454,2.641368643483525,2.419502251829988,-3.2654552986344143,-3.0023322853469634,2.0,2.0,2.0,2.0,2.0
41.0,2.0,1.0,14.0,4.408414181539802,1.1969408910886905,2.0,2.0
29.0,2.0,1.0,14.0,4.254862588769548,3.7418654708604198,2.0,2.0
21.0,2.0,1.0,14.0,4.71986007777415,4.784718884162019,2.0,2.0
"""

"""
Analysis
"""

foldername = 'bin2'
target_Genes = [55, 48, 12, 53, 41, 29, 21]


number_sc = 2500
number_train = 2000
number_bins = 2
number_test = number_sc - number_train

groundtruth = np.loadtxt('SERGIO/steady-state_input_GRN.txt', dtype=np.str)
ground_temp = np.array([each.split(',')[:2] for each in groundtruth]).astype(np.float)


# load test adataset
X = np.loadtxt('SERGIO/scale_2500sc.txt', dtype=float)
X_test = pd.DataFrame(X[(number_train*number_bins):, :], columns= range(X.shape[1]))
X_test = X_test.drop(target_Genes, axis=1)

key_TF = pd.DataFrame(columns=['target_gene', 'ground truth'])

for g, target_gene in enumerate(target_Genes):
    # load json and create model
    # load_name = '|'.join(cells_test) + '@' + target_gene
    save_dir = "sergio_results/%s" % (foldername)
    save_name = 'sergio@%d' % (target_gene)
    json_file = open("%s/generator_%s.json" % (save_dir, save_name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    Generator = model_from_json(loaded_model_json)
    # load weights into new model
    Generator.load_weights("%s/weights_%s.h5" % (save_dir, save_name))

    loss = pd.read_pickle("%s/loss_%s.pkl" % (save_dir, save_name))
    print("Loaded model from disk")


    # 3. Get the selection probability on the testing set
    # retriving the ground truth
    temp = ground_temp[:, 0] == target_gene
    truth_index = int(ground_temp[:, 1][temp])
    truth = groundtruth[temp][0].split(',')[2:(2+truth_index)]

    Sel_Prob_Test = np.asarray(Generator.predict(X_test))
    inferred_index = list()

    for bin in range(number_bins):
        score = np.asarray(1. * (Sel_Prob_Test[range(number_test*bin, number_test*(bin+1))] > 0.5))
        inferred_index.append(np.asarray(np.where(score.sum(0)/len(score)>0.6)))

    key_TF = key_TF.append({'target_gene': np.str(target_gene),
                            'infered key TFs in bin 0': X_test.columns[inferred_index[0]].tolist(),
                            'infered key TFs in bin 1': X_test.columns[inferred_index[1]].tolist(),
                            'ground truth': truth},
                           ignore_index=True)


    n = loss.shape[0]

    for i, name in enumerate(loss.columns):
        plt.scatter(range(n), loss[name], s = 5)
    plt.legend(loc=1, framealpha=1, fontsize=8)
        plt.title('Target gene:%s' % (target_gene))
    #plt.suptitle('Target gene:%s' % (target_gene), fontsize=16)
    plt.show()

    """   
    
    key_TF = key_TF.append({'target_gene': np.str(target_gene),
                        'infered key TFs in bin 0': inferred_index[0],
                        'infered key TFs in bin 1': inferred_index[1],
                        'ground truth': truth},
                       ignore_index=True)
                       """

#fig.tight_layout()
#fig.subplots_adjust(top=.1)


xx = pd.DataFrame(loss, columns=['d_loss', 'v_loss', 'g_loss'])
xx.to_pickle("%s/loss_%s.pkl" % (save_dir, save_name))



"""
Trying 

"""

xx = Sel_Prob_Test.std(axis = 0)/Sel_Prob_Test.shape[0]
xx.size
score = np.asarray(1. * (Sel_Prob_Test > 0.6))




target_Gene = 48
Y_temp = X[:, target_Gene]
Y_scale = np.interp(Y_temp, (Y_temp.min(), Y_temp.max()), (0, +1)).reshape(-1, 1)
Y_train = np.concatenate((Y_scale, 1 - Y_scale), 1)

plt.hist(Y_temp)
plt.show()
plt.hist(Y_temp/Y_temp.max())
plt.show()

input_file_regs = 'SERGIO/steady-state_input_MRs.txt'
with open(input_file_regs, 'r') as f:
    masterRegs = []
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        print(np.shape(row)[0])
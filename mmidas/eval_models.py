import numpy as np
import pickle
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.special import softmax
from torch.utils.data import DataLoader, TensorDataset


def summarize_inference(cpl_mixVAE, files, data, saving_folder=''):
    
    """
        Inference summary for the cpl_mixVAE model

    input args
        cpl_mixVAE: the cpl_mixVAE class object.
        files: the list of model files to be evaluated.
        data: the input data loader.
        saving_folder: the path to save the output dictionary.

    return
        data_dic: the output dictionary containing the summary of the inference.
    """

    n_arm = cpl_mixVAE.n_arm
    n_categories = cpl_mixVAE.n_categories

    data_set_torch = torch.FloatTensor(data['log1p'])
    data_ind_torch = torch.FloatTensor(np.arange(len(data['cluster_id'])))
    torch_data = TensorDataset(data_set_torch, data_ind_torch)
    data_loader = DataLoader(torch_data, batch_size=1000, shuffle=True, drop_last=False, pin_memory=True)


    recon_loss = []
    label_pred = []
    test_dist_c = []
    test_dist_qc = []
    n_pruned = []
    consensus_min = []
    consensus_mean = []
    cT_vs_cT = []
    test_loss = [[] for arm in range(n_arm)]
    prune_indx = []
    consensus = []
    AvsB = []
    sample_id = []
    data_rec = []

    if not isinstance(files, list):
        files = [files]

    for i, file in enumerate(files):
        file_name_ind = file.rfind('/')
        print(f'Model {file[file_name_ind:]}')
        cpl_mixVAE.load_model(file)
        output_dict = cpl_mixVAE.eval_model(data_loader)

        x_low = output_dict['x_low']
        predicted_label = output_dict['predicted_label']
        test_dist_c.append(output_dict['total_dist_z'])
        test_dist_qc.append(output_dict['total_dist_qz'])
        recon_loss.append(output_dict['total_loss_rec'])
        c_prob = output_dict['z_prob']
        prune_indx.append(output_dict['prune_indx'])
        sample_id.append(output_dict['data_indx'])
        label_pred.append(predicted_label)

        category_vs_class = np.zeros((n_arm, data['n_type'], n_categories))

        for arm in range(n_arm):
            test_loss[arm].append(output_dict['total_loss_rec'][arm])
            label_predict = []
            for d in range(len(data['cluster_id'])):
                z_cat = np.squeeze(c_prob[arm][d, :])
                category_vs_class[arm, int(data['cluster_id'][d] - 1), np.argmax(z_cat)] += 1

        if cpl_mixVAE.ref_prior:
            n_arm += 1

        for arm_a in range(n_arm):
            pred_a = predicted_label[arm_a, :]
            for arm_b in range(arm_a + 1, n_arm):
                pred_b = predicted_label[arm_b, :]
                armA_vs_armB = np.zeros((n_categories, n_categories))

                for samp in range(pred_a.shape[0]):
                    armA_vs_armB[pred_a[samp].astype(int) - 1, pred_b[samp].astype(int) - 1] += 1

                num_samp_arm = []
                for ij in range(n_categories):
                    sum_row = armA_vs_armB[ij, :].sum()
                    sum_column = armA_vs_armB[:, ij].sum()
                    num_samp_arm.append(max(sum_row, sum_column))

                armA_vs_armB_norm = np.divide(armA_vs_armB, np.array(num_samp_arm), out=np.zeros_like(armA_vs_armB),
                                         where=np.array(num_samp_arm) != 0)
                nprune_indx = np.where(np.isin(range(n_categories), prune_indx[i]) == False)[0]
                armA_vs_armB_norm = armA_vs_armB_norm[:, nprune_indx][nprune_indx]
                armA_vs_armB = armA_vs_armB[:, nprune_indx][nprune_indx]
                diag_term = np.diag(armA_vs_armB_norm)
                ind_sort = np.argsort(diag_term)
                consensus_min.append(np.min(diag_term))
                con_mean = 1. - (sum(np.abs(predicted_label[0, :] - predicted_label[1, :]) > 0.) / predicted_label.shape[1])
                consensus_mean.append(con_mean)
                AvsB.append(armA_vs_armB)
                consensus.append(armA_vs_armB_norm)

        n_pruned.append(len(nprune_indx))
        category_vs_class = category_vs_class[:, :, nprune_indx]
        cT_vs_cT.append(category_vs_class)
        plt.close()

    data_dic = {}
    data_dic['recon_loss'] = test_loss
    data_dic['dc'] = test_dist_c
    data_dic['d_qc'] = test_dist_qc
    data_dic['con_min'] = consensus_min
    data_dic['con_mean'] = consensus_mean
    data_dic['num_pruned'] = n_pruned
    data_dic['pred_label'] = label_pred
    data_dic['cT_vs_cT'] = cT_vs_cT
    data_dic['consensus'] = consensus
    data_dic['armA_vs_armB'] = AvsB
    data_dic['prune_indx'] = prune_indx
    data_dic['nprune_indx'] = nprune_indx
    data_dic['state_mu'] = output_dict['state_mu']
    data_dic['state_sample'] = output_dict['state_sample']
    data_dic['state_var'] = output_dict['state_var']
    data_dic['sample_id'] = sample_id
    data_dic['c_prob'] = c_prob
    data_dic['lowD_x'] = x_low
    data_dic['x_rec'] = data_rec

    if len(saving_folder) > 0:
        f_name = saving_folder + '/summary_performance_K_' + str(n_categories) + '_narm_' + str(n_arm) + '.p'
        f = open(f_name, "wb")
        pickle.dump(data_dic, f)
        f.close()

    return data_dic
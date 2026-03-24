import numpy as np
from sklearn.preprocessing import normalize
import pickle
import matplotlib.pyplot as plt
import pdb


def summarize_inference(cpl_mixVAE, files, data, saving_folder='', min_c_size=0, verbose=True):
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

    # Initialize lists to store metrics
    recon_loss = []           # Reconstruction loss
    label_pred = []           # Predicted labels
    test_dist_c = []          # Total distance of z to cluster centers
    test_dist_qc = []         # Total distance of q(z) to cluster centers
    n_pruned = []             # Number of pruned categories
    consensus_min = []        # Minimum consensus value
    consensus_mean = []       # Mean consensus value
    test_loss = [[] for a in range(cpl_mixVAE.n_arm)]         # Test loss for each arm
    prune_indx = []           # Pruned indices
    consensus = []            # Consensus matrix
    AvsB = []                 # ArmA vs. ArmB comparison
    sample_id = []            # Sample IDs
    data_rec = []             # Reconstructed data

    # Check if `files` is a list
    if not isinstance(files, list):
        files = [files]

    # Loop over the models and evaluate the performance
    for i, file in enumerate(files):
        # Extract model name from file path
        file_name_ind = file.rfind('/')
        if verbose:
            print(f'Model {file[file_name_ind:]}')

        # Load the model
        cpl_mixVAE.load_model(file)
        
        # Evaluate the model
        output_dict = cpl_mixVAE.eval_model(data)

        # Extract evaluation results
        x_low = output_dict['x_low']
        predicted_label = output_dict['predicted_label']
        test_dist_c.append(output_dict['total_dist_z'])
        test_dist_qc.append(output_dict['total_dist_qz'])
        recon_loss.append(output_dict['total_loss_rec'])
        c_prob = output_dict['z_prob']
        prune_indx.append(output_dict['prune_indx'])
        sample_id.append(output_dict['data_indx'])
        label_pred.append(predicted_label)
        update_prun = False

        # Store test loss for each arm
        for arm in range(cpl_mixVAE.n_arm):
            test_loss[arm].append(output_dict['total_loss_rec'][arm])

        # Adjust number of arms if reference prior is used
        if cpl_mixVAE.ref_prior:
            cpl_mixVAE.n_arm += 1

        # Calculate consensus values
        for arm_a in range(cpl_mixVAE.n_arm):
            pred_a = predicted_label[arm_a, :]
            add_prun_index_a = []
            for cc in np.unique(pred_a):
                if sum(pred_a == cc) < min_c_size:
                    update_prun = True
                    add_prun_index_a.append(int(cc - 1))
                    sorted_c = np.argsort(c_prob[arm_a][pred_a == cc, :], axis=1)
                    updated_c = np.array(list(map(lambda c: select_valid(prune_indx[i], c), sorted_c)))
                    predicted_label[arm_a, pred_a == cc]= updated_c + 1
                    
            pred_a = predicted_label[arm_a, :]  
            add_prun_index_b = []      
            for arm_b in range(arm_a + 1, cpl_mixVAE.n_arm):
                pred_b = predicted_label[arm_b, :]
                for cc in np.unique(pred_a):
                    if sum(pred_b == cc) < min_c_size:
                        update_prun = True
                        add_prun_index_b.append(int(cc - 1))
                        sorted_c = np.argsort(c_prob[arm_b][pred_b == cc, :], axis=1)
                        updated_c = np.array(list(map(lambda c: select_valid(prune_indx[i], c), sorted_c)))
                        predicted_label[arm_b, pred_b == cc]= updated_c + 1
                
                pred_b = predicted_label[arm_b, :]   
                armA_vs_armB = np.zeros((cpl_mixVAE.n_categories, cpl_mixVAE.n_categories))

                for samp in range(pred_a.shape[0]):
                    armA_vs_armB[pred_a[samp].astype(int) - 1, pred_b[samp].astype(int) - 1] += 1

                armA_vs_armB_norm = normalize(armA_vs_armB, axis=1, norm='l1')
                if update_prun:
                    add_prun_index = set(add_prun_index_a).union(set(add_prun_index_b))
                    prune_indx[i] = np.concatenate((prune_indx[i], np.array(list(add_prun_index))))
                
                nprune_indx = np.where(np.isin(range(cpl_mixVAE.n_categories), prune_indx[i]) == False)[0]
                armA_vs_armB_norm = armA_vs_armB_norm[:, nprune_indx][nprune_indx]
                armA_vs_armB = armA_vs_armB[:, nprune_indx][nprune_indx]
                diag_term = np.diag(armA_vs_armB_norm)
                ind_sort = np.argsort(diag_term)
                consensus_min.append(np.min(diag_term))
                # con_mean = 1. - (sum(np.abs(pred_a != pred_b)) / predicted_label.shape[1])
                con_mean = np.mean(diag_term)
                consensus_mean.append(con_mean)
                AvsB.append(armA_vs_armB)
                consensus.append(armA_vs_armB_norm)

        # Store number of pruned indices
        n_pruned.append(len(nprune_indx))
        plt.close()
        # pdb.set_trace()

    # Create dictionary to store results
    data_dic = {}
    data_dic['recon_loss'] = test_loss
    data_dic['dc'] = test_dist_c
    data_dic['d_qc'] = test_dist_qc
    data_dic['con_min'] = consensus_min
    data_dic['con_mean'] = consensus_mean
    data_dic['num_pruned'] = n_pruned
    data_dic['pred_label'] = label_pred
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

    # Save dictionary to file if saving_folder is provided
    if len(saving_folder) > 0:
        f_name = saving_folder + '/summary_performance_K_' + str(cpl_mixVAE.n_categories) + '_narm_' + str(cpl_mixVAE.n_arm) + '.p'
        f = open(f_name, "wb")
        pickle.dump(data_dic, f)
        f.close()

    return data_dic



def select_valid(prune_index, c):
    i = 1
    while np.isin(c[-1 - i], prune_index):
        i += 1
        
    return c[-1 - i]
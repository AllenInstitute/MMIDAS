import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def RF_classifier(data, labels, kfold, seed):
    """
    Random Forest classifier

    Args:
        data (np.array): data
        labels (dict): labels
        kfold (int): number of folds
        seed (int): random seed

    Returns:
        acc (dict): accuracy
        ref_labels (dict): reference labels
        pred_labels (dict): predicted labels
    """

    kf = KFold(n_splits=kfold, random_state=seed, shuffle=True)
    acc = dict()
    pred_labels = dict()
    ref_labels = dict()

    for ik, key in enumerate(labels.keys()):
        y = labels[key]
        acc[key] = []
        pred_labels[key] = []
        ref_labels[key] = []

        for train_index, test_index in kf.split(data):
            rfc = RandomForestClassifier()
            rfc.fit(data[train_index, :], y[train_index])
            y_pred = rfc.predict(data[test_index, :])
            acc[key].append(accuracy_score(y[test_index], y_pred))
            pred_labels[key].append(y_pred)
            ref_labels[key].append(y[test_index])

    return acc, ref_labels, pred_labels


def LDA_classifier(data, labels, kfold, seed):
    """
    Linear Discriminant Analysis classifier

    Args:
        data (np.array): data
        labels (dict): labels
        kfold (int): number of folds
        seed (int): random seed

    Returns:
        acc (dict): accuracy
        ref_labels (dict): reference labels
        pred_labels (dict): predicted labels

    """

    kf = KFold(n_splits=kfold, random_state=seed, shuffle=True)
    acc = dict()
    pred_labels = dict()
    ref_labels = dict()

    for ik, key in enumerate(labels.keys()):
        y = labels[key]
        acc[key] = []
        pred_labels[key] = []
        ref_labels[key] = []

        for train_index, test_index in kf.split(data):
            lda = LinearDiscriminantAnalysis(store_covariance=True)
            lda.fit(data[train_index, :], y[train_index])
            y_pred = lda.predict(data[test_index, :])
            acc[key].append(accuracy_score(y[test_index], y_pred))
            pred_labels[key].append(y_pred)
            ref_labels[key].append(y[test_index])

    return acc, ref_labels, pred_labels


def QDA_classifier(data, labels, kfold, seed):
    """
    Quadratic Discriminant Analysis classifier

    Args:
        data (np.array): data
        labels (dict): labels
        kfold (int): number of folds
        seed (int): random seed

    Returns:
        acc (dict): accuracy
        ref_labels (dict): reference labels
        pred_labels (dict): predicted labels

    """

    kf = KFold(n_splits=kfold, random_state=seed, shuffle=True)
    acc = dict()
    pred_labels = dict()
    ref_labels = dict()

    for ik, key in enumerate(labels.keys()):
        y = labels[key]
        acc[key] = []
        pred_labels[key] = []
        ref_labels[key] = []

        for train_index, test_index in kf.split(data):
            qda = QuadraticDiscriminantAnalysis(reg_param=1e-2, store_covariance=True)
            qda.fit(data[train_index, :], y[train_index])
            y_pred = qda.predict(data[test_index, :])
            acc[key].append(accuracy_score(y[test_index], y_pred))
            pred_labels[key].append(y_pred)
            ref_labels[key].append(y[test_index])

    return acc, ref_labels, pred_labels



def cluster_compare(data, labels, num_pc=0, saving_path=''):
    """
    Compare clustering results

    Args:
        data (np.array): data
        labels (dict): labels
        num_pc (int): number of PCs
        saving_path (str): path to save the figure

    Returns:
        fig (plt.figure): figure
        silh_smp_score (list): silhouette sample score
        sil_score (list): silhouette score
        c_size (list): cluster size

    """

    fig = plt.figure(figsize=[10, 5])
    ax = fig.add_subplot()

    if num_pc > 0:
        pca = PCA(n_components=num_pc)
        z = pca.fit(data).transform(data)
        silh_smp_score, sil_score = [], []
        c_size = []
        for ik, key in enumerate(labels.keys()):
            y = labels[key]
            uni_class = np.unique(y)
            sample_score = silhouette_samples(z, y)
            sil_score.append(silhouette_score(z, y))
            mean_smp_sc = np.zeros(len(uni_class))
            cluster_size = np.zeros(len(uni_class))
            for ic, c in enumerate(np.unique(y)):
                label_ind = np.where(labels[key] == c)[0]
                mean_smp_sc[ic] = np.mean(sample_score[label_ind])
                cluster_size[ic] = len(label_ind)

            silh_smp_score.append(mean_smp_sc)
            sort_indx = np.argsort(mean_smp_sc)
            c_size.append(cluster_size[sort_indx])
            ax.plot(np.arange(len(uni_class)), mean_smp_sc[sort_indx], label=key)

        ax.set_title(str(num_pc) + ' PCs', fontsize=18)
        ax.set_xlabel('Ordered clusters')
        ax.legend(prop={'size': 12})
        ax.set_ylabel('Ave. Silhouette scores')
        fig.tight_layout()

    return fig, silh_smp_score, sil_score, c_size


def K_selection(data_dict, num_category, n_arm, thr=0.95):
    """
    Select the number of clusters

    Args:
        data_dict (dict): data dictionary
        num_category (int): number of categories
        n_arm (int): number of arms
        thr (float): threshold for consensus value

    Returns:
        data_dict['num_pruned'][indx]: number of pruned indices
        l_recon_mean[indx]: mean reconstruction loss
        consensus[indx]: consensus
        K: number of clusters   
    """

    n_comb = max(n_arm * (n_arm - 1) / 2, 1)

    with sns.axes_style("darkgrid"):
        data_dict['num_pruned'] = np.array(data_dict['num_pruned'])
        data_dict['dc'] = np.array(data_dict['dc'])
        data_dict['d_qc'] = np.array(data_dict['d_qc'])
        data_dict['con_min'] = np.array(data_dict['con_min'])
        data_dict['con_min'] = np.reshape(data_dict['con_min'], (int(n_comb), len(data_dict['d_qc'])))
        data_dict['con_mean'] = np.array(data_dict['con_mean'])
        data_dict['con_mean'] = np.reshape(data_dict['con_mean'], (int(n_comb), len(data_dict['d_qc'])))
        indx = np.argsort(data_dict['num_pruned'])
        norm_aitchison_dist = data_dict['dc'] - np.min(data_dict['dc'])
        norm_aitchison_dist = norm_aitchison_dist / np.max(norm_aitchison_dist)
        recon_loss = []
        norm_recon = []
        l_recon = []

        for a in range(n_arm):
            recon_loss.append(np.array(data_dict['recon_loss'][a]))
            # print(np.min(recon_loss[a]),  np.max(recon_loss[a]))
            tmp = recon_loss[a] - np.min(recon_loss[a])
            norm_recon.append(tmp / np.max(tmp))
            l_recon.append(recon_loss[a])

        norm_recon_mean = np.mean(norm_recon, axis=0)
        l_recon_mean = np.mean(l_recon, axis=0)
        neg_cons = 1 - np.mean(data_dict['con_mean'], axis=0)
        consensus = np.mean(data_dict['con_mean'], axis=0)
        mean_cost = (neg_cons + norm_recon_mean + norm_aitchison_dist) / 3 # cplmixVAE_data['d_qz']
        
        # suggest the number of clusters
        if thr > max(consensus):
            print("Required minimum consensus is set too high, kindly consider specifying a lower value.")
            plot_flag = False
            K = None
        else:
            plot_flag = True
            ordered_rec = l_recon_mean[indx]
            ordered_cons = consensus[indx]
            tmp_ind = np.where(ordered_cons > thr)[0]
            max_changes_indx = np.where(np.diff(ordered_cons[tmp_ind]) == max(np.diff(ordered_cons[tmp_ind])))[0][0] + 1
            selected_idx = max_changes_indx
            K = data_dict['num_pruned'][indx][selected_idx] 
            
            # for tt in range(len(tmp_ind)):
            #     i = len(tmp_ind) - tt - 1
            #     if (ordered_cons[tmp_ind[i]] > ordered_cons[tmp_ind[i]-1]) and (ordered_rec[tmp_ind[i]] < ordered_rec[tmp_ind[i]-1]):
            #         selected_idx = tmp_ind[i]  
            #         K = data_dict['num_pruned'][indx][selected_idx] 
            #         break
        
        fig = plt.figure(figsize=[10, 5])
        ax = fig.add_subplot()
        ax.plot(data_dict['num_pruned'][indx], data_dict['d_qc'][indx], label='Average Distance')
        ax.plot(data_dict['num_pruned'][indx], neg_cons[indx], label='Average Dissent (1 - Consensus)')
        ax.set_xlim([np.min(data_dict['num_pruned'][indx])-1, num_category + 1])
        ax.set_xlabel('Categories', fontsize=14)
        ax.set_xticks(data_dict['num_pruned'][indx])
        ax.set_xticklabels(data_dict['num_pruned'][indx], fontsize=8, rotation=90)
        y_max = np.max([np.max(data_dict['d_qc']), np.max(neg_cons)]) + 0.1
        if plot_flag:
            ax.vlines(data_dict['num_pruned'][indx][selected_idx], 0, y_max, colors='gray', linestyles='dotted')
            ax.hlines(neg_cons[indx][selected_idx], min(data_dict['num_pruned']), max(data_dict['num_pruned']), colors='gray', linestyles='dotted')
        
        ax.legend(loc='upper right')
        ax.set_ylim([0, y_max])
        ax.grid(True)
        plt.show()

        if plot_flag:
            print(f"Selected number of clusters: {data_dict['num_pruned'][indx][selected_idx]} with consensus {consensus[indx][selected_idx]}")

        return data_dict['num_pruned'][indx], l_recon_mean[indx], consensus[indx], K



def get_SilhScore(x, labels):
    """
    Calculate silhouette score

    Args:
        x (np.array): data
        labels (np.array): labels

    Returns:
        mean_smp_sc (np.array): mean silhouette sample score
        sil_score (float): silhouette score
    """

    uni_class = np.unique(labels)
    sample_score = silhouette_samples(x, labels)
    sil_score = silhouette_score(x, labels)
    mean_smp_sc = np.zeros(len(uni_class))
    for ic, c in enumerate(uni_class):
        label_ind = np.where(labels == c)[0]
        mean_smp_sc[ic] = np.mean(sample_score[label_ind])

    return mean_smp_sc, sil_score
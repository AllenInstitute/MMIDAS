import argparse
import os, pickle
import numpy as np
import scanpy as sc
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.cluster import DBSCAN
from utils.config import load_config
from utils.dataloader import get_adata, load_data
from utils.data_tools import split_data_Kfold
from utils.cluster_analysis import get_SilhScore

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--n_gene", default=5000, type=int, help="number of genes")
parser.add_argument("--n_dim", default=100, type=int, help="number of dimensions aftter dimension reduction")
parser.add_argument("--n_neighbors", default=10, type=int, help="number of neighbors for KNN")
parser.add_argument("--resolution", default=5, type=float, help="A parameter value controlling the coarseness of the clustering. Higher values lead to more clusters.")
parser.add_argument("--n_iteration", default=-1, type=int, help="number of iterations for Leiden algorithm")
parser.add_argument("--mode", default='linear', type=str, help="dimension reduction method, linear or nlinear")
parser.add_argument("--K_fold", default=10, type=int, help="number of folds for cross validation")
parser.add_argument("--clustering", default='leiden', type=str, help="clustering method, leiden, louvain, or DB")
parser.add_argument("--training", default=True, type=bool, help="Enable training mode")


# Main function
def main(n_dim, n_neighbors, resolution, K_fold, n_iteration, clustering, n_gene, mode, training):

    # Load paths from configuration file
    paths = load_config(config_file='config.toml')
    saving_folder = paths['package_dir'] / paths['saving_folder_seurat']
    data_file_count = paths['package_dir'] / paths['data_count']
    data_file = paths['package_dir'] / paths['data']
    htree_file = paths['package_dir'] / paths['htree_file']

    folder_name = f'method_{clustering}_{mode}_nDim_{n_dim}_resolution_{resolution}_nKNN_{n_neighbors}'
    saving_folder = str(saving_folder)

    # Load data
    data_count = load_data(file=data_file_count, measure='counts', n_gene=n_gene, ref_genes=True)
    data = load_data(file=data_file, measure='log1p', n_gene=n_gene, ref_genes=True)

    # Perform dimension reduction and clustering, including linear and non-linear methods
    if training:
        if mode == 'linear':
            # obtain PCA representation
            adata = get_adata(data=data, measure=data_count['log1p'])
            sc.pp.pca(adata, n_comps=n_dim)
            sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X_pca')
            pca = PCA(n_components=n_dim)
            z = pca.fit(data['log1p']).transform(data['log1p'])
        else:
            # Load the latent space obtained from scVI
            z = np.load(saving_folder + f'/Z_scvi_10.npy', allow_pickle=True)
            adata = get_adata(data=data_count, measure=z)
            sc.pp.neighbors(adata, n_neighbors=n_neighbors)

        # Cluster the data
        print(f'{clustering} clustering')
        if clustering == 'leiden':
            sc.tl.leiden(adata, resolution=resolution, random_state=0, n_iterations=n_iteration, key_added='leiden', directed=False)
            label_T = adata.obs['leiden'].values
        elif clustering == 'DB':
            db_cluster = DBSCAN(eps=resolution, min_samples=n_neighbors).fit(z)
            label_T = db_cluster.labels_

        n_type = len(np.unique(label_T))
        print(f'Number of clusters: {n_type}')

        # Save clustering results
        file = saving_folder + f"/{folder_name}.p"
        f = open(file, "wb")
        sum_dict = dict()
        sum_dict['sample_id'] = data['sample_id']
        sum_dict['cluster'] = label_T
        sum_dict['n_class'] = n_type
        pickle.dump(sum_dict, f, protocol=4)
        f.close()
    else:
        if mode == 'linear':
            pca = PCA(n_components=n_dim)
            z = pca.fit(data['log1p']).transform(data['log1p'])
        else:
            z = np.load(saving_folder + f'/Z_scvi_10.npy', allow_pickle=True)

        # Load clustering results
        file = saving_folder + f"/{folder_name}.p"
        with open(file, 'rb') as f:
            sum_dict = pickle.load(f)
        label_T = sum_dict['cluster']
        n_type = sum_dict['n_class']

    # Evaluate clustering performance
    sc_T, _ = get_SilhScore(z,  label_T)
    train_ind, test_ind = split_data_Kfold(label_T, K_fold)
    acc_T = []
    acc_T_adj = []
    pred_label = []
    true_label = []
    tlabel = []
    for fold in range(K_fold):
        print(f'------ fold: {fold} ------')
        train_id = train_ind[fold].astype(int)
        test_id = test_ind[fold].astype(int)
        rfc = RandomForestClassifier()
        rfc.fit(z[train_id, :], label_T[train_id])
        y_pred = rfc.predict(z[test_id, :])
        pred_label.append(y_pred)
        true_label.append(label_T[test_id])
        acc_T.append(accuracy_score(label_T[test_id], y_pred))
        acc_T_adj.append(balanced_accuracy_score(label_T[test_id], y_pred))
        tlabel.append(data['cluster'][test_id])
        print(f"{acc_T_adj[-1]}")

    # Print average performance metrics
    print(f"Average performance: {np.mean(acc_T_adj)}, {np.std(acc_T_adj)}, {np.mean(sc_T)}, {np.std(sc_T)}")

    # Concatenate true and predicted labels for confusion matrix
    true_label = np.concatenate(true_label)
    pred_label = np.concatenate(pred_label)
    conf_mat = confusion_matrix(true_label, pred_label, normalize='true')
    tlabel = np.concatenate(tlabel)

    # Save results to file
    unique_label = np.unique(true_label)
    T_class_ord = []
    for label in unique_label:
        T_class_ord.append(tlabel[true_label == label][0])

    data_file_id = saving_folder + f"/Ttype_classification_{mode}_{clustering}_K_{n_type}_nFeature_{n_dim}.p"
    f = open(data_file_id, "wb")
    sum_dict = dict()
    sum_dict['acc_T'] = acc_T
    sum_dict['acc_T_adj'] = acc_T_adj
    sum_dict['n_class'] = n_type
    sum_dict['conf_mat'] = conf_mat
    sum_dict['cluster'] = label_T
    sum_dict['ttypes'] = data['cluster']
    sum_dict['sc_T'] = sc_T
    pickle.dump(sum_dict, f, protocol=4)
    f.close()


if __name__ == "__main

import pickle
import glob
import numpy as np
import scanpy as sc
import igraph as ig
import leidenalg as la
from sklearn.ensemble import RandomForestClassifier
import anndata
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from scipy.sparse import csc_matrix
from utils.config import load_config
from utils.dataloader import get_adata, load_data
from utils.data_tools import split_data_Kfold
from utils.cluster_analysis import get_SilhScore

# Load paths from configuration file
paths = load_config(config_file='config.toml')
saving_folder = paths['package_dir']
data_file_count = paths['package_dir'] / paths['data_count']
data_file = paths['package_dir'] / paths['data']
saving_folder = str(saving_folder)

# Load data from file
file = glob.glob(saving_folder + '/clustering_study/ACTIONet/*normalized*.h5ad')[0]
print(file)
adata = anndata.read_h5ad(file)
#data_count = load_data(file=data_file_count, measure='counts', n_gene=5000, ref_genes=True)
# data = load_data(file=data_file, measure='log1p', n_gene=n_gene, ref_genes=True)

# Extract relevant data
G_matrix = adata.obsp["ACTIONet"]
z = adata.obsm["H_stacked"].todense()
print(z.shape)
del adata

# Convert the adjacency matrix to an igraph Graph object
nonzero_indices = np.transpose(G_matrix.nonzero())
G = ig.Graph(edges=nonzero_indices.tolist(), directed=False, edge_attrs={'weight': G_matrix.data})

# Cluster the graph using Leiden algorithm
resolution_parameter = 0.04
partition = la.find_partition(G, la.CPMVertexPartition, resolution_parameter=resolution_parameter)

# Get cluster labels
label_T = np.array(partition.membership)
n_type = len(np.unique(label_T))
n_dim = z.shape[-1]
print(f'Number of clusters: {n_type}')

# Perform K-fold cross-validation
K_fold = 10
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
    print(f"{acc_T_adj[-1]}")

print(f"Average performance: {np.mean(acc_T_adj)}, {np.std(acc_T_adj)}, {np.mean(sc_T)}, {np.std(sc_T)}")

# Concatenate true and predicted labels for confusion matrix
true_label = np.concatenate(true_label)
pred_label = np.concatenate(pred_label)
conf_mat = confusion_matrix(true_label, pred_label, normalize='true')

# Save results to file
data_file_id = saving_folder + f"/clustering_study/ACTIONet/Ttype_classification_ACTIONet_K_{n_type}_nFeature_{n_dim}.p"
f = open(data_file_id, "wb")
sum_dict = dict()
sum_dict['acc_T'] = acc_T
sum_dict['acc_T_adj'] = acc_T_adj
sum_dict['n_class'] = n_type
sum_dict['conf_mat'] = conf_mat
sum_dict['cluster'] = label_T
sum_dict['sc_T'] = sc_T
pickle.dump(sum_dict, f, protocol=4)
f.close()
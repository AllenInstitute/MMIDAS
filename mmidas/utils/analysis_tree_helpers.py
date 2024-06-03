import fnmatch
import os
import pprint

import feather
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy


class Node():
    '''Simple Node class. Each instance contains a list of children and parents.'''

    def __init__(self, name, C_list=[], P_list=[]):
        self.name = name
        self.C_name_list = C_list[P_list == name]
        self.P_name = P_list[C_list == name]
        return

    def __repr__(self):
        # Invoked when printing a list of Node objects
        return self.name

    def __str__(self):
        # Invoked when printing a single Node object
        return self.name

    def __eq__(self, other):

        if isinstance(other, self.__class__):
            return self.name == other.name
        else:
            return False

    def children(self, C_list=[], P_list=[]):
        return [Node(n, C_list, P_list) for n in self.C_name_list]

def get_descendants(child, parent, y, ancestor, leafonly=False):
    '''Return a list consisting of all descendents for a given ancestor. \n
    `leafonly=True` returns only leaf node descendants'''

    descendant = []
    visitnode = child[parent == ancestor].tolist()
    descendant.extend(visitnode)
    while visitnode:
        ancestor = visitnode.pop(0)
        nextgen = child[parent == ancestor].tolist()
        visitnode.extend(nextgen)
        descendant.extend(nextgen)

    # Exclude non leaf descendants if leafonly=True. Leaf nodes wil never appear as parents.
    if leafonly:
        descendant = list(set(descendant).difference(set(parent)))

    return descendant

def get_ancestors(node, child, parent):
    '''Return a list consisting of all ancestors for a given cell type. \n
    `leafonly=True` returns only leaf node descendants'''

    ancestor = []
    visitnode = node
    while visitnode != 'n1':
        visitnode = parent[child == visitnode]
        ancestor.append(visitnode[0])

    return ancestor


def get_merge_sequence(child, parent, y):
    '''Returns `list_changes` consisting of \n
    1. list of children to merge \n
    2. parent label to merge the children into \n
    3. number of remaining nodes in the tree'''

    # Log changes for every merge step
    remaining_nodes = np.unique(child).shape[0]
    list_changes = []
    y[y == 0] = np.nan
    while remaining_nodes > 1:
        # Find lowest hanging parent node
        minind = np.nanargmin(y)
        this_parent = child[minind]

        # Find children of this_parent to merge - there can be an arbitrary number of children.
        c_ind = np.where(parent == this_parent)[0]
        child_list = child[c_ind].tolist()

        # Remove merged children from the data
        child = np.delete(child, c_ind)
        parent = np.delete(parent, c_ind)
        y = np.delete(y, c_ind)

        y[child == this_parent] = np.nan
        remaining_nodes = np.unique(child).shape[0]
        list_changes.append([child_list, this_parent, remaining_nodes])

    return list_changes


def do_merges(labels, list_changes=[], n_merges=0):
    '''Perform n_merges on an array of labels using the list of changes at each merge.'''

    for i in range(n_merges):
        if i < len(list_changes):
            c_nodes_list = list_changes[i][0]
            p_node = list_changes[i][1]
            for c_node in c_nodes_list:
                n_samples = np.sum([labels == c_node])
                labels[labels == c_node] = p_node
                # print(n_samples,' in ',c_node, ' --> ' ,p_node)
        else:
            print('Exiting after performing max allowed merges =',
                  len(list_changes))
            break
    return labels


def parse_dend(htree_file):
    '''Parses the 'dend' file to output \n
    `list_changes`: (see module get_merge_sequence) \n
    `descendants`: dict with node labels as keys and all list of descendants as values
    `leaves`: numpy array of all leaf node names
    ``
    '''
    # The htree_file is extracted from the corresponding dend.RData file.
    # R functions `dend_functions.R` and `dend_parents.R` are used for this (Ref. Rohan/Zizhen)
    # y and height variables have the same values
    treeobj = pd.read_csv(htree_file)
    treeobj = treeobj[['x', 'y', 'leaf', 'label', 'parent', 'col']]
    treeobj['leaf'] = treeobj[
                          'leaf'].values == True  # Contains nan values otherwise
    treeobj = treeobj.sort_values(by=['y', 'x'], axis=0,
                                  ascending=[True, True]).copy(deep=True)
    treeobj = treeobj.reset_index(drop=True).copy(deep=True)
    treeobj['y'].values[treeobj['leaf'].values] = np.nan

    child = treeobj['label'].values
    parent = treeobj['parent'].values
    y = treeobj['y'].values
    leaves = child[treeobj['leaf'].values]
    list_changes = get_merge_sequence(child, parent, y)

    # Create a dictionary to list all descendants for a given node (key)
    descendants = {}
    ancestor_list = [x for x in list(set(parent).union(set(child))) if
                     str(x) != 'nan']
    for p in ancestor_list:
        descendants[p] = get_descendants(child, parent, y, p, leafonly=False)

    return list_changes, descendants, treeobj, leaves, child, parent


def plot_htree(htree_file):
    """
    Plots the hierarchy tree from the htree_file. 
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()

    _, _, treeobj, _, child, parent = parse_dend(htree_file)

    xx = treeobj['x'].values
    yy = treeobj['y'].values
    yy[np.isnan(yy)] = 0
    isleaf = treeobj['leaf'].values == True
    col = treeobj['col'].values
    col[~isleaf] = '#000000'

    fig = plt.figure(figsize=(15, 10))
    for i, s in enumerate(child):
        col[i]
        plt.text(xx[i], yy[i], s, horizontalalignment='center',
                 verticalalignment='top', rotation=90, color=col[i], fontsize=8)

    for p in parent:
        xp = xx[child == p]
        yp = yy[child == p]
        ch = child[parent == p]
        for c in ch:
            xc = xx[child == c]
            yc = yy[child == c]
            plt.plot([xc, xc], [yc, yp], color='#BBBBBB')
            plt.plot([xc, xp], [yp, yp], color='#BBBBBB')

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([np.min(xx) - 1, np.max(xx) + 1])
    ax.set_ylim([-0.0, 1.05])
    return


def get_cvfold(cvfile='', refdata={}):
    '''Loads training and validation data from a particular cross validation set. \n
    Returns `paired` data and `leaf` data dictionaries with fields `T_z`,`E_z`,`color`, `labels` and `labels_id`.'''

    cvfile = sio.loadmat(cvfile, squeeze_me=True)

    # Find corresponding metadata for training set
    T_train_z = cvfile['z_train_0']
    E_train_z = cvfile['z_train_1']
    train_color = refdata['cluster_color'][cvfile['train_ind_T']]
    train_labels = refdata['cluster'][cvfile['train_ind_T']]
    train_labels_id = refdata['clusterID'][cvfile['train_ind_T']]

    T_train_ispaired = refdata['T_ispaired'][cvfile['train_ind_T']] == 1
    E_train_ispaired = refdata['E_ispaired'][cvfile['train_ind_E']] == 1

    # restrict training dataset to only the paired samples
    T_train_paired_z = T_train_z[T_train_ispaired, :]
    E_train_paired_z = E_train_z[E_train_ispaired, :]
    train_paired_color = train_color[T_train_ispaired]
    train_paired_labels = train_labels[T_train_ispaired]
    train_paired_labels_id = train_labels_id[T_train_ispaired]

    # Find corresponding metadata for validation set. All validation samples are paired by design.
    T_val_z = cvfile['z_val_0']
    E_val_z = cvfile['z_val_1']
    val_color = refdata['cluster_color'][cvfile['val_ind']]
    val_labels = refdata['cluster'][cvfile['val_ind']]
    val_labels_id = refdata['clusterID'][cvfile['val_ind']]

    train_paired = {'T_z': T_train_paired_z,
                    'E_z': E_train_paired_z,
                    'color': train_paired_color,
                    'labels': train_paired_labels,
                    'labels_id': train_paired_labels_id}

    val_paired = {'T_z': T_val_z,
                  'E_z': E_val_z,
                  'color': val_color,
                  'labels': val_labels,
                  'labels_id': val_labels_id}

    # Restrict GMM fitting using only leaf nodes, even for merged clusters
    train_isleaf = train_paired_color != '#808080'
    T_train_leaf_z = T_train_paired_z[train_isleaf]
    E_train_leaf_z = E_train_paired_z[train_isleaf]
    train_leaf_color = train_paired_color[train_isleaf]
    train_leaf_labels = train_paired_labels[train_isleaf]
    train_leaf_labels_id = train_paired_labels_id[train_isleaf]

    # Restrict validation set to transcriptomic leaf nodes:
    val_isleaf = val_color != '#808080'
    T_val_leaf_z = T_val_z[val_isleaf]
    E_val_leaf_z = E_val_z[val_isleaf]
    val_leaf_color = val_color[val_isleaf]
    val_leaf_labels = val_labels[val_isleaf]
    val_leaf_labels_id = val_labels_id[val_isleaf]

    train_leaf = {'T_z': T_train_leaf_z,
                  'E_z': E_train_leaf_z,
                  'color': train_leaf_color,
                  'labels': train_leaf_labels,
                  'labels_id': train_leaf_labels_id}

    val_leaf = {'T_z': T_val_leaf_z,
                'E_z': E_val_leaf_z,
                'color': val_leaf_color,
                'labels': val_leaf_labels,
                'labels_id': val_leaf_labels_id}

    return train_paired, val_paired, \
           train_leaf, val_leaf


def get_cvfold_extended(cvfile='', refdata={}, full_data=False):
    '''Loads training and validation data from a particular cross validation set. \n
    Returns `paired` data and `leaf` data dictionaries.'''

    cvfile = sio.loadmat(cvfile, squeeze_me=True)

    # Find corresponding metadata for training set
    T_train_z = cvfile['z_train_0']
    E_train_z = cvfile['z_train_1']
    train_color = refdata['cluster_color'][cvfile['train_ind_T']]
    train_labels = refdata['cluster'][cvfile['train_ind_T']]
    train_labels_id = refdata['clusterID'][cvfile['train_ind_T']]

    T_train_ind = cvfile['train_ind_T'].copy()
    E_train_ind = cvfile['train_ind_E'].copy()

    T_train_ispaired = refdata['T_ispaired'][cvfile['train_ind_T']] == 1
    E_train_ispaired = refdata['E_ispaired'][cvfile['train_ind_E']] == 1

    # restrict training dataset to only the paired samples
    T_train_paired_z = T_train_z[T_train_ispaired, :]
    E_train_paired_z = E_train_z[E_train_ispaired, :]
    train_paired_color = train_color[T_train_ispaired]
    train_paired_labels = train_labels[T_train_ispaired]
    train_paired_labels_id = train_labels_id[T_train_ispaired]

    T_train_paired_ind = T_train_ind[T_train_ispaired]
    E_train_paired_ind = E_train_ind[E_train_ispaired]

    # Find corresponding metadata for validation set. All validation samples are paired by design.
    T_val_z = cvfile['z_val_0']
    E_val_z = cvfile['z_val_1']
    val_color = refdata['cluster_color'][cvfile['val_ind']]
    val_labels = refdata['cluster'][cvfile['val_ind']]
    val_labels_id = refdata['clusterID'][cvfile['val_ind']]
    T_val_paired_ind = cvfile['val_ind'].copy()
    E_val_paired_ind = cvfile['val_ind'].copy()

    train_paired = {'T_z': T_train_paired_z,
                    'E_z': E_train_paired_z,
                    'color': train_paired_color,
                    'labels': train_paired_labels,
                    'labels_id': train_paired_labels_id,
                    'T_ind': T_train_paired_ind,
                    'E_ind': E_train_paired_ind}

    val_paired = {'T_z': T_val_z,
                  'E_z': E_val_z,
                  'color': val_color,
                  'labels': val_labels,
                  'labels_id': val_labels_id,
                  'T_ind': T_val_paired_ind,
                  'E_ind': E_val_paired_ind}

    # Restrict GMM fitting using only leaf nodes, even for merged clusters
    train_isleaf = train_paired_color != '#808080'
    T_train_leaf_z = T_train_paired_z[train_isleaf]
    E_train_leaf_z = E_train_paired_z[train_isleaf]
    train_leaf_color = train_paired_color[train_isleaf]
    train_leaf_labels = train_paired_labels[train_isleaf]
    train_leaf_labels_id = train_paired_labels_id[train_isleaf]

    T_train_leaf_ind = T_train_paired_ind[train_isleaf]
    E_train_leaf_ind = E_train_paired_ind[train_isleaf]

    # Restrict validation set to transcriptomic leaf nodes:
    val_isleaf = val_color != '#808080'
    T_val_leaf_z = T_val_z[val_isleaf]
    E_val_leaf_z = E_val_z[val_isleaf]
    val_leaf_color = val_color[val_isleaf]
    val_leaf_labels = val_labels[val_isleaf]
    val_leaf_labels_id = val_labels_id[val_isleaf]

    T_val_leaf_ind = T_val_paired_ind[val_isleaf]
    E_val_leaf_ind = E_val_paired_ind[val_isleaf]

    train_leaf = {'T_z': T_train_leaf_z,
                  'E_z': E_train_leaf_z,
                  'color': train_leaf_color,
                  'labels': train_leaf_labels,
                  'labels_id': train_leaf_labels_id,
                  'T_ind': T_train_leaf_ind,
                  'E_ind': E_train_leaf_ind}

    val_leaf = {'T_z': T_val_leaf_z,
                'E_z': E_val_leaf_z,
                'color': val_leaf_color,
                'labels': val_leaf_labels,
                'labels_id': val_leaf_labels_id,
                'T_ind': T_val_leaf_ind,
                'E_ind': E_val_leaf_ind}

    if full_data:
        # Return original gene expression and feature values for this fold
        train_leaf['T_x'] = refdata['T_dat'][train_leaf['T_ind'], :]
        train_leaf['E_x'] = refdata['E_dat'][train_leaf['E_ind'], :]
        val_leaf['T_x'] = refdata['T_dat'][val_leaf['T_ind'], :]
        val_leaf['E_x'] = refdata['E_dat'][val_leaf['E_ind'], :]

    return train_paired, val_paired, \
           train_leaf, val_leaf


def custom_QDA(train_z, true_train_lbl, test_z, true_test_lbl,
               n_per_class_thr=6, diag_cov_n_sample_thr=12):
    '''Supervised fitting of gaussians to classes independently on training set and prediction of class membership on test set. \n
    Classes are not weighted, i.e. in: p(class|z) ~ p(z|class)*p(class), p(class) is assumed to be 1 \n
    This function has not been tested well.'''
    from scipy.stats import multivariate_normal as mvn

    # Supervised GMM fitting: Fit a gaussian to samples of each label
    lbl_name = []
    lbl_mean = []
    lbl_cov = []
    excluded_lbl = []

    unique_lbl = np.unique(np.concatenate([true_train_lbl, true_test_lbl]))
    pred_test_pdfval = np.empty((test_z.shape[0], unique_lbl.size))
    pred_test_pdfval.fill(0)

    for i, lbl in enumerate(unique_lbl):
        this_z = train_z[true_train_lbl == lbl, :]
        if this_z.shape[0] > n_per_class_thr:
            lbl_name.append(lbl)
            lbl_mean.append(np.mean(this_z, axis=0))
            cov = np.cov(this_z, rowvar=False)
            if this_z.shape[0] < diag_cov_n_sample_thr:
                lbl_cov.append(np.diagonal(cov))
            else:
                lbl_cov.append(cov)

            pred_test_pdfval[:, i] = mvn.pdf(test_z, lbl_mean[-1], lbl_cov[-1])
        else:
            excluded_lbl.append(lbl)

    best_inds = np.argmax(pred_test_pdfval, axis=1)
    pred_test_lbl = unique_lbl[best_inds]

    # Exclude labels with insufficient n(samples)
    if true_test_lbl:
        for lbl in excluded_lbl:
            pred_test_lbl[true_test_lbl == lbl] = 'excluded'
            true_test_lbl[true_test_lbl == lbl] = 'excluded'

    return true_test_lbl, pred_test_lbl


def predict_leaf_gmm(train_z, true_train_lbl, test_z, true_test_lbl=[],
                     n_per_class_thr=6, diag_cov_n_sample_thr=12,
                     unique_dataset_lbl=[], unique_leaf_lbl=[],
                     descendant_dict={}, label_weight=[]):
    '''Assign new labels using Gaussian fits to the training samples for each unique label in true_train_lbl.\n
    If `descendant_dict` is provided, label predictions are obtained by \n
    1. first combining probabilities of all descendant labels \n
    2. then choosing the maximum among the combined labels \n
    3. `unique_dataset_lbl` is a list of labels that remain after merging'''
    import pdb
    import numpy as np
    from scipy.stats import multivariate_normal as mvn

    if descendant_dict:
        assert unique_dataset_lbl, 'unique_dataset_lbl should not be empty if descendant_dict is provided'

    # Supervised GMM fitting: Fit a gaussian to samples of each label
    lbl_name = []
    lbl_mean = []
    lbl_cov = []
    excluded_lbl = []

    unique_dataset_lbl = np.array(unique_dataset_lbl)
    unique_leaf_lbl = np.unique(unique_leaf_lbl)

    pred_test_pdfval = np.empty((test_z.shape[0], unique_leaf_lbl.size))
    pred_test_pdfval.fill(0)

    # Obtain n_test_cells x leaf_labels prediction matrix
    for i, lbl in enumerate(unique_leaf_lbl):
        this_z = train_z[true_train_lbl == lbl, :]
        if this_z.shape[0] > n_per_class_thr:
            lbl_name.append(lbl)
            lbl_mean.append(np.mean(this_z, axis=0))
            cov = np.cov(this_z, rowvar=False)
            cov = cov + 1e-4 * np.eye(this_z.shape[1])
            if this_z.shape[0] < diag_cov_n_sample_thr:
                cov = np.diagonal(cov)
            lbl_cov.append(cov)

            # Predict probability for all test cells to have this label.
            pred_test_pdfval[:, i] = mvn.pdf(test_z, lbl_mean[-1], lbl_cov[-1])
        else:
            excluded_lbl.append(lbl)
    if label_weight:
        pred_test_pdfval = np.multiply(pred_test_pdfval,
                                       np.reshape(label_weight, (1, -1)))
    pred_test_pdfval = np.divide(pred_test_pdfval,
                                 np.sum(pred_test_pdfval, axis=1,
                                        keepdims=True))

    # Remove test cells that have an excluded true_test_lbl
    keep = np.invert(np.isin(true_test_lbl, excluded_lbl))
    n_excluded_cells = np.sum(np.invert(keep))
    test_z = test_z[keep, :]
    true_test_lbl = true_test_lbl[keep]
    pred_test_pdfval = pred_test_pdfval[keep, :]

    # Create map from current labels to merged labels
    from_label = []
    to_label = []
    for key, val in descendant_dict.items():
        if val:
            for v in val:
                from_label.extend([v])
                to_label.extend([key])

    from_label = np.array(from_label)
    to_label = np.array(to_label)

    # Initialize merged probability matrix
    pred_test_pdfval_merged = np.empty(
        (test_z.shape[0], unique_dataset_lbl.size))
    pred_test_pdfval_merged.fill(0)

    for i, this_label in enumerate(unique_dataset_lbl):

        if this_label in unique_leaf_lbl:
            pred_test_pdfval_merged[:, i] = np.squeeze(
                pred_test_pdfval[:, unique_leaf_lbl == this_label])
        elif to_label.size > 0:
            merge_these = from_label[to_label == this_label]
            merge_inds = np.isin(unique_leaf_lbl, merge_these)
            if np.sum(merge_inds) != 0:
                pred_test_pdfval_merged[:, i] = np.sum(
                    pred_test_pdfval[:, merge_inds], axis=1)
        else:
            pdb.set_trace()

    best_inds = np.argmax(pred_test_pdfval_merged, axis=1)
    pred_test_lbl = unique_dataset_lbl[best_inds]
    pred_probability = np.max(pred_test_pdfval_merged, axis=1)

    # Assign merged labels to each cell
    for k in range(len(from_label)):
        true_test_lbl[true_test_lbl == from_label[k]] = to_label[k]

    n_classes_removed = np.sum(np.isin(unique_dataset_lbl, excluded_lbl))
    n_classes_predicted = np.sum(
        np.invert(np.isin(unique_dataset_lbl, excluded_lbl)))
    return true_test_lbl, pred_test_lbl, n_excluded_cells, n_classes_predicted, n_classes_removed, pred_probability


def get_cca_projections(train_leaf, val_leaf, n_components=3):
    '''Obtain training and validation set latent space coordinates.\n
    dicts `train_leaf` and `val_leaf` must have paired data in keys `T_x` and `E_x` \n
    The low dimensional projections are whitened to make the scale the same. \n'''
    from scipy.linalg import sqrtm
    from sklearn.cross_decomposition import CCA

    # Dim reduction with CCA
    this_CCA = CCA(n_components=n_components, scale=True, max_iter=1e4,
                   tol=1e-06, copy=True)
    this_CCA.fit(train_leaf['T_x'], train_leaf['E_x'])

    # Whiten cca projections
    train_cca = {}
    val_cca = {}
    train_cca['T_z'], train_cca['E_z'] = this_CCA.transform(train_leaf['T_x'],
                                                            train_leaf['E_x'])
    val_cca['T_z'], val_cca['E_z'] = this_CCA.transform(val_leaf['T_x'],
                                                        val_leaf['E_x'])

    for key, X in train_cca.items():
        X = X - np.mean(X, axis=0)
        train_cca[key] = np.matmul(X, sqrtm(
            np.linalg.inv(np.cov(np.transpose(X)))))

    for key, X in val_cca.items():
        X = X - np.mean(X, axis=0)
        val_cca[key] = np.matmul(X,
                                 sqrtm(np.linalg.inv(np.cov(np.transpose(X)))))

    for key in ['color', 'labels', 'labels_id']:
        train_cca[key] = train_leaf[key].copy()
        val_cca[key] = val_leaf[key].copy()

    return train_cca, val_cca
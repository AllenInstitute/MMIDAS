import numpy as np
import scanpy as sc
import seaborn as sns
from sklearn.preprocessing import normalize
import matplotlib.colors as mcolors
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import pdb
import json
from scipy.sparse import issparse
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



def normalize_cellxgene(x):
    """Normalize based on number of input genes

    Args:
        x (np.array): cell x gene matrix (cells along axis=0, genes along axis=1)
        scale_factor (float): Scalar multiplier
    
    Returns: 
        x, np.mean(x)
    """
    # x = np.divide(x, np.sum(x, axis=1, keepdims=True))*scale_factor

    return normalize(x, axis=1, norm='l1')



def logcpm(x, scaler=1e4) -> np.array:
    """ Log CPM normalization

    inpout args
        x (np.array): cell x gene matrix (cells along axis=0, genes along axis=1)
        scaler (float, optional): scaling factor for log CPM
    
    return 
        normalized log CPM gene expression matrix
    """
    return np.log1p(normalize_cellxgene(x) * scaler)



def sparse_std(x, axis=None):
    x_ = x.copy()
    x_.data **= 2
    return np.sqrt(x_.mean(axis) - np.square(x.mean(axis)))



def reorder_genes(x, eps, chunksize=1000):
    t_gene = x.shape[1]
    g_std = []
    if issparse(x):
        g_std = sparse_std(x, axis=0)
    else:
        g_std = np.std(x, axis=0)

    # for iter in range(int(t_gene // chunksize) + 1):
    #     ind0 = iter * chunksize
    #     ind1 = np.min((t_gene, (iter + 1) * chunksize))
    #     # x_bin = np.where(x[:, ind0:ind1] > eps, 1, 0)
    #     g_std.append(np.std(x[:, ind0:ind1], axis=0))
    #     g_bin_std.append(np.std(x_bin, axis=0))

    # g_std = np.concatenate(g_std)
    # g_bin_std = np.concatenate(g_bin_std)
    g_ind = np.argsort(g_std)
    g_ind = g_ind[np.sort(g_std) > eps]
    return g_ind[::-1]



def get_HVG(x, thr=0.1, binary=True):
    if binary:
        x_bin = (x != 0).astype(int)
        g_index = reorder_genes(x_bin, thr)
    else:
        g_index = reorder_genes(x, thr)
    
    return np.asarray(g_index).flatten()



def tfidf(x, scaler=1e4):
    tfidf = TfidfTransformer(norm='l1', use_idf=True)
    return tfidf.fit_transform(x) * scaler



def get_HAP(x, thr=0.1, binary=True):
    if binary:
        x_bin = (x != 0).astype(int)
        colsum = np.array(np.sum(x_bin, axis=0))
    else:
        colsum = np.array(np.sum(x, axis=0))

    colsum = colsum.reshape(-1)
    return np.logical_and((colsum > x.shape[0] * thr), (colsum < x.shape[0] * (1 - thr)))


def generate_colors(n):
    # Generate `n` unique colors using the `viridis` colormap
    # if n <= 10:  # The colorblind palette has 10 colors
    #     palette = sns.color_palette("colorblind", n)
    # else:
    palette = sns.color_palette("hsv", n)  # Use HSV for larger n
    colors = [mcolors.rgb2hex(color) for color in palette]
    
    return np.array(colors)


def split_data_Kfold(class_label, K_fold):
    uniq_label = np.unique(class_label)
    label_train_indices = [[] for ll in uniq_label]
    label_test_indices = [[] for ll in uniq_label]

    # Split the the data to train and test keeping the same ratio for all classes
    for i_l, label in enumerate(uniq_label):
        label_indices = np.where(class_label == label)[0]
        test_size = int(( 1 /K_fold) * len(label_indices))

        # Prepare the test and training indices for K folds
        for fold in range(K_fold):
            ind_0 = fold * test_size
            ind_1 = (1 + fold) * test_size
            tmp_ind = list(label_indices)
            label_test_indices[i_l].append(tmp_ind[ind_0:ind_1])
            del tmp_ind[ind_0:ind_1]
            label_train_indices[i_l].append(tmp_ind)
    test_ind = [[] for k in range(K_fold)]
    train_ind = [[] for k in range(K_fold)]
    for fold in range(K_fold):
        for i_l in range(len(uniq_label)):
            test_ind[fold].append(label_test_indices[i_l][fold])
            train_ind[fold].append(label_train_indices[i_l][fold])
        test_ind[fold] = np.concatenate(test_ind[fold])
        train_ind[fold] = np.concatenate(train_ind[fold])
        # Shuffle the indices
        index = np.arange(len(test_ind[fold]))
        np.random.shuffle(index)
        test_ind[fold] = test_ind[fold][index]
        index = np.arange(len(train_ind[fold]))
        np.random.shuffle(index)
        train_ind[fold] = train_ind[fold][index]

    return train_ind, test_ind



def load_data(file, gene_file='', n_gene=0):

    if gene_file:
        df_ = pd.read_csv(gene_file)
        for key in df_.keys():
            # check key include gene in the name
            if 'gene' in key.lower():
                gene_list = df_[key].values
                break
    data = dict()     
    if isinstance(file, list):
        X = []
        for f in file:
            adata = sc.read_h5ad(f)
            genes = adata.var.index.values
            g_index = [np.where(genes == g)[0][0] for g in gene_list if g in genes]
            X.append(adata.X[:, g_index].toarray())
        
        try:
            data['log1p'] = np.vstack(X)
            data['gene_id'] = gene_list
        except:
            print("--------> Cannot combine dataset! <--------")
    else:
        adata = sc.read_h5ad(file)
        genes = adata.var.index.values
        if gene_file:
            g_index = [np.where(genes == g)[0][0] for g in gene_list if g in genes]
            data['log1p'] = adata.X[:, g_index].toarray()
            data['gene_id'] = gene_list
        else:
            data['log1p'] = adata.X.toarray()
            data['gene_id'] = np.hstack(adata.var.values)
    
    if n_gene > 0:
        data['log1p'] = data['log1p'][:, :n_gene]
        data['gene_id'] = data['gene_id'][:n_gene]
    
    if not isinstance(file, list):
        for key in adata.obs.keys():
            data[key] = adata.obs[key].values
            if key == 'sex':
                data[key] = np.array([s.split(';')[0] for s in data[key]])
            
    print(f"Number of cells: {data['log1p'].shape[0]}, Number of genes: {data['log1p'].shape[1]}")

    return data



def get_data(x, train_size, additional_val, seed):

        test_size = x.shape[0] - train_size
        train_cpm, test_cpm, train_ind, test_ind = train_test_split(x, np.arange(x.shape[0]), train_size=train_size, test_size=test_size, random_state=seed)
        
        if additional_val:
            train_cpm, val_cpm, train_ind, val_ind = train_test_split(train_cpm, train_ind, train_size=train_size - test_size, test_size=test_size, random_state=seed)
        else:
            val_cpm = []
            val_ind = []

        return train_cpm, val_cpm, test_cpm, train_ind, val_ind, test_ind



def get_loaders(x, label=[], batch_size=128, train_size=0.9, n_aug_smp=0, netA=None, additional_val=False, seed=0):

    if len(label) > 0:
        train_ind, val_ind, test_ind = [], [], []
        for ll in np.unique(label):
            indx = np.where(label == ll)[0]
            tt_size = int(train_size * sum(label == ll))
            _, _, _, train_subind, val_subind, test_subind = get_data(x, tt_size, additional_val, seed)
            train_ind.append(indx[train_subind])
            test_ind.append(indx[test_subind])
            if additional_val:
                val_ind.append(indx[val_subind])

        train_ind = np.concatenate(train_ind)
        test_ind = np.concatenate(test_ind)
        train_set = x[train_ind, :]
        test_set = x[test_ind, :]
        
        if additional_val:
            val_ind = np.concatenate(val_ind)
            val_set = x[val_ind, :]
        
    else:
        tt_size = int(train_size * x.shape[0])
        train_set, val_set, test_set, train_ind, val_ind, test_ind = get_data(x, tt_size, additional_val, seed)

    train_set_torch = torch.FloatTensor(train_set)
    train_ind_torch = torch.FloatTensor(train_ind)
    if n_aug_smp > 0:
        train_set = train_set_torch.clone()
        train_set_ind = train_ind_torch.clone()
        for _ in range(n_aug_smp):
            if netA:
                _, fake_data = netA(train_set_torch, True, .1)
                train_set = torch.cat((train_set, fake_data.cpu().detach()), 0)

            else:
                train_set = torch.cat((train_set, train_set_torch), 0)
                
            train_set_ind = torch.cat((train_set_ind, train_ind_torch), 0)

        train_data = TensorDataset(train_set, train_set_ind)
    else:
        train_data = TensorDataset(train_set_torch, train_ind_torch)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)

    test_set_torch = torch.FloatTensor(test_set)
    test_ind_torch = torch.FloatTensor(test_ind)
    test_data = TensorDataset(test_set_torch, test_ind_torch)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True, drop_last=False, pin_memory=True)

    data_set_troch = torch.FloatTensor(x)
    all_ind_torch = torch.FloatTensor(range(x.shape[0]))
    all_data = TensorDataset(data_set_troch, all_ind_torch)
    alldata_loader = DataLoader(all_data, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)
    
    if additional_val:
        val_set_torch = torch.FloatTensor(val_set)
        val_ind_torch = torch.FloatTensor(val_ind)
        validation_data = TensorDataset(val_set_torch, val_ind_torch)
        validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True)
    else:
        validation_loader = []
        val_ind = []

    return alldata_loader, train_loader, test_loader, validation_loader, test_ind, val_ind


def Dbh_Retro_loaders(x_Dbh, x_Retro, label=[], batch_size=128, train_size=0.9, n_aug_smp=0, netA=None, additional_val=False, seed=0):

    tt_size = int(train_size * x_Dbh.shape[0])
    train_set_Dbh, val_set_Dbh, test_set_Dbh, train_ind_Dbh, val_ind_Dbh, test_ind_Dbh = get_data(x_Dbh, tt_size, additional_val, seed)

    train_set_torch_Dbh = torch.FloatTensor(train_set_Dbh)
    train_ind_torch_Dbh = torch.FloatTensor(train_ind_Dbh)
    if n_aug_smp > 0:
        train_set_Dbh = train_set_torch_Dbh.clone()
        train_set_ind_Dbh = train_ind_torch_Dbh.clone()
        train_set_Dbh_aug = []
        for _ in range(n_aug_smp):
            if netA:
                _, fake_data = netA(train_set_Dbh, True, .1)
                train_set_Dbh_aug.append(fake_data.cpu().detach())
                del fake_data
            else:
                train_set_Dbh_aug.append(train_set_Dbh)            
        
        train_set_torch_Dbh = torch.cat((train_set_Dbh, torch.cat((train_set_Dbh_aug), dim=0)), dim=0)
        train_ind_torch_Dbh = train_set_ind_Dbh.repeat(n_aug_smp + 1)
        del train_set_Dbh, train_set_ind_Dbh
        print(f'New size of Dbh: {train_set_torch_Dbh.shape[0]}')
    
    if len(label) > 0:
        train_ind, val_ind, test_ind = [], [], []
        for ll in np.unique(label):
            indx = np.where(label == ll)[0]
            tt_size = int(train_size * sum(label == ll))
            _, _, _, train_subind, val_subind, test_subind = get_data(x_Retro[indx], tt_size, additional_val, seed)
            train_ind.append(indx[train_subind])
            test_ind.append(indx[test_subind])
            if additional_val:
                val_ind.append(indx[val_subind])

        train_ind_retro = np.concatenate(train_ind)
        test_ind_retro = np.concatenate(test_ind)
        train_set_retro = x_Retro[train_ind_retro]
        test_set_retro = x_Retro[test_ind_retro]
        
        if additional_val:
            val_ind_retro = np.concatenate(val_ind)
            val_set_retro = x_Retro[val_ind_retro]
        
    else:
        train_set_retro = x_Retro.copy()
        train_ind_retro = range(x_Retro.shape[0])
        val_set_retro = x_Retro.copy()
        val_ind_retro = range(x_Retro.shape[0])
        test_set_retro = x_Retro.copy()
        test_ind_retro = range(x_Retro.shape[0])

    train_set_torch_retro = torch.FloatTensor(train_set_retro)
    train_ind_torch_retro = torch.FloatTensor(train_ind_retro)
    if n_aug_smp > 0:
        train_set_retro = train_set_torch_retro.clone()
        train_set_ind_retro = train_ind_torch_retro.clone()
        n_aug_smp_ = n_aug_smp * int(x_Dbh.shape[0]/ x_Retro.shape[0])
        print(f'Number of augmentation samples for RetroSeq: {n_aug_smp_}')
        train_set_retro_aug = []
        for _ in range(n_aug_smp_):
            if netA:
                _, fake_data = netA(train_set_retro, True, .1)
                train_set_retro_aug.append(fake_data.cpu().detach())
                del fake_data
            else:
                train_set_retro_aug.append(train_set_retro)
                
        
        train_set_torch_retro = torch.cat((train_set_retro, torch.cat((train_set_retro_aug), dim=0)), dim=0)
        train_ind_torch_retro = train_set_ind_retro.repeat(n_aug_smp_ + 1)
        del train_set_retro, train_set_ind_retro
        
        print(f'New size of retro: {train_set_torch_retro.shape[0]}')
    
    train_data = TensorDataset(torch.cat((train_set_torch_Dbh, train_set_torch_retro), dim=0), torch.cat((train_ind_torch_Dbh, train_ind_torch_retro), dim=0))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)

    test_set_torch = torch.FloatTensor(np.concatenate((test_set_Dbh, test_set_retro), axis=0))
    test_ind_torch = torch.FloatTensor(np.concatenate((test_ind_Dbh, test_ind_retro), axis=0))
    test_data = TensorDataset(test_set_torch, test_ind_torch)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True, drop_last=False, pin_memory=True)

    data_set_troch = torch.FloatTensor(np.concatenate((x_Dbh, x_Retro), axis=0))
    all_ind_torch = torch.FloatTensor(range(x_Dbh.shape[0] + x_Retro.shape[0]))
    all_data = TensorDataset(data_set_troch, all_ind_torch)
    alldata_loader = DataLoader(all_data, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)
    
    if additional_val:
        val_set_torch = torch.FloatTensor(np.concatenate((val_set_Dbh, val_set_retro), axis=0))
        val_ind_torch = torch.FloatTensor(np.concatenate((val_ind_Dbh, val_ind_retro), axis=0))
        validation_data = TensorDataset(val_set_torch, val_ind_torch)
        validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True)
    else:
        validation_loader = []
        val_ind_torch = []
        
    return alldata_loader, train_loader, test_loader, validation_loader, test_ind_torch, val_ind_torch



def select_best_genes(X, y, n_top_genes=500, K_fold=10, n_repeat=10):
    
    mutual_features = []
    for repeat in range(n_repeat):
        train_ind, test_ind = split_data_Kfold(y, K_fold)
        collect_features = []
        for fold in range(K_fold):
            print(f"------- {repeat} - {fold} -------")
            train_id = train_ind[fold].astype(int)
            test_id = test_ind[fold].astype(int)
            X_train, X_test = X[train_id, :], X[test_id, :]
            y_train, y_test = y[train_id], y[test_id]
            clf = RandomForestClassifier()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy_all = accuracy_score(y_test, y_pred)

            # Get feature importance scores
            feature_importances = clf.feature_importances_
            # Select top `n_top_genes`
            top_gene_indices = np.argsort(feature_importances)[::-1][:n_top_genes]
            collect_features.append(top_gene_indices)

            # Train and evaluate with selected genes
            X_train_sel, X_test_sel = X_train[:, top_gene_indices], X_test[:, top_gene_indices]
            clf = RandomForestClassifier()
            clf.fit(X_train_sel, y_train)
            y_pred = clf.predict(X_test_sel)
            accuracy_select = accuracy_score(y_test, y_pred)
            print(f"Accuracy before feature selection: {accuracy_all}")
            print(f"Accuracy after feature selection: {accuracy_select}")
        
        mutual_features.append(np.array(list(set.intersection(*(set(ff) for ff in collect_features)))))
        clf = RandomForestClassifier()
        clf.fit(X[:, mutual_features[-1]], y)
        y_pred = clf.predict(X[:, mutual_features[-1]])
        accuracy_select = accuracy_score(y, y_pred)
        print(f"Overall accuracy after feature selection: {accuracy_select}, for {len(mutual_features[-1])} genes")
    
    final_features = np.array(list(set.union(*(set(ff) for ff in mutual_features))))
    clf = RandomForestClassifier()
    clf.fit(X[:, final_features], y)
    y_pred = clf.predict(X[:, final_features])
    accuracy_select = accuracy_score(y, y_pred)
    print(f"Overall accuracy after feature selection: {accuracy_select}, for {len(final_features)} genes")
    
    return final_features



def jaccard_distance(target: torch.Tensor, prediction: torch.Tensor, scaled: bool = True) -> torch.Tensor:
    """
    Compute the Jaccard distance between two binary tensors.
    
    Args:
        target (torch.Tensor): Target tensor.
        prediction (torch.Tensor): Prediction tensor.
        
    Returns:
        torch.Tensor: Jaccard distance.
    """
    # Ensure binary tensors
    y_true = target.bool() if hasattr(target, "bool") else target
    y_pred = prediction.bool() if hasattr(prediction, "bool") else prediction
    
    # Compute intersection and union
    intersection = torch.sum(y_true & y_pred)
    union = torch.sum(y_true | y_pred)
    
    # Jaccard Index and Distance
    jaccard_index = intersection / union if union > 0 else torch.tensor(0.0)
    if scaled:
        return (1 - jaccard_index) * union
    else:
        return 1 - jaccard_index


def load_mixed_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:  # Ensure it's not an empty line
                try:
                    parsed_line = json.loads(line)  # Parse JSON (dict or value)
                except json.JSONDecodeError:
                    parsed_line = line  # Keep as string if it's not valid JSON
                data.append(parsed_line)
    return data
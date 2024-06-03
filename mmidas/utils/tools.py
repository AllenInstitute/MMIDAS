import numpy as np
import scipy.io as sio
import toml
from pathlib import Path
from sklearn.preprocessing import normalize

def get_paths(toml_file, sub_file='files', verbose=False):
    """Loads dictionary with path names and any other variables set through xxx.toml

    Args:
        verbose (bool, optional): print paths

    Returns:
        config: dict
    """

    package_dir = Path().resolve().parents[1]
    config_file = package_dir / toml_file
    print(config_file)

    if not Path(config_file).is_file():
        print(f'Did not find project`s toml file: {config_file}')

    f = open(config_file, "r")
    config = toml.load(f)
    f.close()

    config['paths'].update({'main_dir': package_dir})

    if verbose:
        for key in config.keys():
            print(f'{key}: {config[key]}')

    for key in config:
        if key=='paths':
            for key2 in config['paths']:
                if Path(config['paths'][key2]).exists():
                    config['paths'][key2] = Path(config['paths'][key2])
        if key==sub_file:
            print(f'Getting files directories belong to {sub_file}...')
            for key2 in config[sub_file]:
                if Path(config[sub_file][key2]).exists():
                    config[sub_file][key2] = Path(config[sub_file][key2])

    return config


def normalize_cellxgene(x) -> np.array:
    """ Normalize based on number of input genes

    inpout args
        x (np.array): cell x gene matrix (cells along axis=0, genes along axis=1)
    
    return
        normalized gene expression matrix
    """
    return normalize(x, axis=1, norm='l1')


def logcpm(x, scaler=1e6) -> np.array:
    """ Log CPM normalization

    inpout args
        x (np.array): cell x gene matrix (cells along axis=0, genes along axis=1)
        scaler (float, optional): scaling factor for log CPM
    
    return 
        normalized log CPM gene expression matrix
    """
    return np.log1p(normalize_cellxgene(x) * scaler)


def print_attrs(name, obj):
    print(name)
    for key, val in obj.attrs.items():
        print("    %s: %s" % (key, val))
    return


def reorder_genes(x, chunksize=1000, eps=1e-1):
    """ Reorder genes based on their variability
    inpout args
        x (np.array): cell x gene matrix (cells along axis=0, genes along axis=1)
        chunksize (int, optional): chunksize for gene reordering
        eps (float, optional): threshold for binarization

    return
        reordered gene indices
    """
    t_gene = x.shape[1]
    print(t_gene)
    g_std, g_bin_std = [], []

    for iter in range(int(t_gene // chunksize) + 1):
        ind0 = iter * chunksize
        ind1 = np.min((t_gene, (iter + 1) * chunksize))
        x_bin = np.where(x[:, ind0:ind1] > eps, 1, 0)
        g_std.append(np.std(x[:, ind0:ind1], axis=0))
        g_bin_std.append(np.std(x_bin, axis=0))

    g_std = np.concatenate(g_std)
    g_bin_std = np.concatenate(g_bin_std)
    g_ind = np.argsort(g_bin_std)
    g_ind = g_ind[np.sort(g_bin_std) > eps]
    print(len(g_ind))
    return g_ind[::-1]


def split_data_Kfold(class_label, K_fold):
    """ Split data into K folds
    inpout args
        class_label (np.array): class labels
        K_fold (int): number of folds

    return
        train_ind, test_ind: indices for training and testing
    """
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

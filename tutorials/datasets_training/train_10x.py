import argparse
import os
from utils.training import train_cplmixVAE
from utils.config import load_config
from utils.dataloader import load_data
from utils.data_tools import reorder_genes
import random
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.special import softmax
import pickle
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import numpy as np


# Setup argument parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--n_categories", default=100, type=int, help="number of cell types, e.g. 120")
parser.add_argument("--state_dim", default=3, type=int, help="state variable dimension")
parser.add_argument("--n_arm", default=2, type=int,  help="number of mixVAE arms for each modalities")
parser.add_argument("--temp",  default=1, type=float, help="gumbel-softmax temperature")
parser.add_argument("--tau",  default=0.005, type=float, help="softmax temperature")
parser.add_argument("--beta",  default=1, type=float, help="KL regularization parameter")
parser.add_argument("--lam",  default=1, type=float, help="coupling factor")
parser.add_argument("--lam_pc",  default=1000, type=float, help="coupling factor for ref arm")
parser.add_argument("--latent_dim", default=10, type=int, help="latent dimension")
parser.add_argument("--n_epoch", default=10000, type=int, help="Number of epochs to train")
parser.add_argument("--n_epoch_p", default=1000, type=int, help="Number of epochs to train pruning algorithm")
parser.add_argument("--min_con", default=.99, type=float, help="minimum consensus")
parser.add_argument("--max_prun_it", default=50, type=int, help="maximum number of pruning iterations")
parser.add_argument("--ref_pc", default=False, type=bool, help="path of the data augmenter")
parser.add_argument("--fc_dim", default=100, type=int, help="number of nodes at the hidden layers")
parser.add_argument("--batch_size", default=1000, type=int, help="batch size")
parser.add_argument("--subclass", default='Vip', type=str, help="cell subclass including gaba and glum")
parser.add_argument("--variational", default=True, type=bool, help="enable variational mode")
parser.add_argument("--augmentation", default=True, type=bool, help="enable VAE-GAN augmentation")
parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
parser.add_argument("--n_gene", default=10000, type=int, help="number of genes")
parser.add_argument("--n_pr", default=0, type=int, help="initial pruning index")
parser.add_argument("--n_zim", default=1, type=int, help="")
parser.add_argument("--p_drop", default=0.5, type=float, help="input probability of dropout")
parser.add_argument("--s_drop", default=0.0, type=float, help="state probability of dropout")
parser.add_argument("--n_run", default=1, type=int, help="number of the experiment")
parser.add_argument("--hard", default=False, type=bool, help="hard encoding")
parser.add_argument("--sort_gene", default=True, type=bool, help="enable sorting genes")
parser.add_argument("--device", default=None, type=int, help="gpu device, use None for cpu")


def main(n_categories, n_arm, state_dim, latent_dim, fc_dim, n_epoch, n_epoch_p, min_con, max_prun_it, batch_size, subclass, sort_gene,
         p_drop, s_drop, lr, temp, n_run, device, hard, tau, variational, ref_pc, augmentation, n_gene, lam, lam_pc, beta, n_pr, n_zim):

    # Load configuration settings from config file
    paths = load_config(config_file='config.toml')
    saving_folder = paths['package_dir'] / paths['saving_folder_cplmix']

    # Load data based on the subclass provided
    if subclass:
        data_file = paths['local_data_path'] / paths['data_' + subclass]
        data = load_data(datafile=data_file) # ref_types=True, ann_smt=paths['ann_smt'], ann_10x=paths['ann_10x'])
        folder_name = f'{subclass}_run_{n_run}_K_{n_categories}_Sdim_{state_dim}_aug_True_nGene_{n_gene}_p_drop_{p_drop}_fc_dim_{fc_dim}_temp_{temp}_' + \
                      f'lr_{lr}_n_arm_{n_arm}_tau_{tau}_lam_{lam}_nbatch_{batch_size}_nepoch_{n_epoch}_nepochP_{n_epoch_p}'

    else:
        data_file_glum = paths['local_data_path'] / paths['data_glum']
        data_file_gaba = paths['local_data_path'] / paths['data_gaba']
        data_gaba = load_data(datafile=data_file_gaba) #, ref_types=True, ann_smt=paths['ann_smt'], ann_10x=paths['ann_10x'])
        data_glum = load_data(datafile=data_file_glum) #, ref_types=True, ann_smt=paths['ann_smt'], ann_10x=paths['ann_10x'])

        data = dict()
        data['gene_id'] = data_gaba['gene_id']
        for key in data_glum.keys():
            if key != 'gene_id':
                print(key)
                data[key] = np.concatenate((data_glum[key], data_gaba[key]))
                print(data[key].shape)

        folder_name = f'run_{n_run}_K_{n_categories}_Sdim_{state_dim}_aug_{augmentation}_p_drop_{p_drop}_fc_dim_{fc_dim}_temp_{temp}_' + \
                      f'lr_{lr}_n_arm_{n_arm}_tau_{tau}_lam_{lam}_nbatch_{batch_size}_nepoch_{n_epoch}_nepochP_{n_epoch_p}'

    # Create directories for saving results
    saving_folder = saving_folder / folder_name
    os.makedirs(saving_folder, exist_ok=True)
    os.makedirs(saving_folder / 'model', exist_ok=True)
    saving_folder = str(saving_folder)

    # Encode cluster labels
    eps = 1e-6
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(data['cluster_order'])
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    data['c_onehot'] = onehot_encoder.fit_transform(integer_encoded)
    data['c_p'] = softmax((data['c_onehot'] + eps) / tau, axis=1)
    data['n_type'] = len(np.unique(data['cluster_order']))

    # Print dataset details
    print(data['log1p'].shape, len(np.unique(data['cluster_label'])))

    # Sort genes if enabled
    if sort_gene:
        g_index = reorder_genes(data['log1p'])
        g_idx = g_index[0::n_arm]
        for arm in range(1, n_arm):
            g_idx = np.concatenate((g_idx, g_index[arm::n_arm]))

    # Load augmentation file if augmentation is enabled
    if augmentation:
        if subclass in ['Vip', 'Pvalb']:
            aug_file = paths['package_dir'] / paths['saving_folder_augmenter'] / paths['aug_file_gaba']
        elif subclass in ['L4-5_IT', 'L2-3_IT']:
            aug_file = paths['package_dir'] / paths['saving_folder_augmenter'] / paths['aug_file_glum']
        else:
            aug_file = paths['package_dir'] / paths['saving_folder_augmenter'] / paths['aug_file_' + subclass]
    else:
        aug_file = ''

    # Select genes based on sorted indices
    data['log1p'] = data['log1p'][:, g_index[:n_gene]]
    data['gene_id'] = data['gene_id'][g_index[:n_gene]]

    print(data['log1p'].shape, len(np.unique(data['cluster_label'])), n_gene)

    # Update number of genes if not specified
    if n_gene == 0:
        n_gene = data['log1p'].shape[1]

    # Initialize and train the coupled mixVAE (MMIDAS) model
    cpl_mixVAE = train_cplmixVAE(saving_folder=saving_folder,
                                 aug_file=aug_file,
                                 device=device,
                                 n_feature=n_gene)

    train_loader, test_loader, alldata_loader, train_ind, test_ind = cpl_mixVAE.getdata(dataset=data['log1p'],
                                                                                      label=data['cluster_order'],
                                                                                      batch_size=batch_size)
    # Initialize model with or without pruning
    if n_pr > 0:
        cpl_mixVAE.init_model(n_categories=n_categories,
                              state_dim=state_dim,
                              input_dim=n_gene,
                              fc_dim=fc_dim,
                              lowD_dim=latent_dim,
                              x_drop=p_drop,
                              s_drop=s_drop,
                              lr=lr,
                              n_arm=n_arm,
                              temp=temp,
                              hard=hard,
                              tau=tau,
                              lam=lam,
                              lam_pc=lam_pc,
                              beta=beta,
                              n_zim=n_zim,
                              variational=variational,
                              ref_prior=ref_pc,
                              n_pr=n_pr,
                              trained_model=saving_folder + str(paths['trained_model']))
    else:
        cpl_mixVAE.init_model(n_categories=n_categories,
                              state_dim=state_dim,
                              input_dim=n_gene,
                              fc_dim=fc_dim,
                              lowD_dim=latent_dim,
                              x_drop=p_drop,
                              s_drop=s_drop,
                              lr=lr,
                              n_arm=n_arm,
                              temp=temp,
                              hard=hard,
                              tau=tau,
                              lam=lam,
                              lam_pc=lam_pc,
                              beta=beta,
                              n_zim=n_zim,
                              variational=variational,
                              ref_prior=ref_pc,
                              n_pr=n_pr)


    cpl_mixVAE.run(train_loader=train_loader,
                   test_loader=test_loader,
                   alldata_loader=alldata_loader,
                   n_epoch=n_epoch,
                   n_epoch_p=n_epoch_p,
                   c_p=data['c_onehot'],
                   min_con=min_con,
                   max_pron_it=max_prun_it,
                   mode='ZINB'
                   )


# Run the main function when the script is executed
if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
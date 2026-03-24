import argparse
import os
from utils.training import train_cplmixVAE
import pickle
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import numpy as np

# Setup argument parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default='/data', type=str, help="Path to the data directory")  # MTG_AD_data/all_donors_data
parser.add_argument("--n_gene", default=9881, type=int, help="Number of genes")
parser.add_argument("--n_categories", default=10, type=int, help="Number of cell types")
parser.add_argument("--state_dim", default=10, type=int, help="State variable dimension")
parser.add_argument("--n_arm", default=2, type=int, help="Number of mixVAE arms for each modality")
parser.add_argument("--temp", default=1, type=float, help="Gumbel-softmax temperature")
parser.add_argument("--tau", default=.1, type=float, help="Softmax temperature")
parser.add_argument("--beta", default=1, type=float, help="Beta factor for regularization")
parser.add_argument("--latent_dim", default=30, type=int, help="Latent dimension")
parser.add_argument("--n_epoch", default=1000, type=int, help="Number of epochs to train")
parser.add_argument("--n_epoch_p", default=1000, type=int, help="Number of epochs to train pruning algorithm")
parser.add_argument("--min_con", default=.99, type=float, help="Minimum consensus for pruning")
parser.add_argument("--max_pron_it", default=8, type=int, help="Maximum number of pruning iterations")
parser.add_argument("--aug_path", default='./results/augmenter', type=str, help="Path to the data augmenter")
parser.add_argument("--fc_dim", default=100, type=int, help="Number of nodes in the hidden layers")
parser.add_argument("--batch_size", default=1000, type=int, help="Batch size")
parser.add_argument("--variational", default=True, type=bool, help="Enable variational mode")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
parser.add_argument("--p_drop", default=0.2, type=float, help="Input probability of dropout")
parser.add_argument("--s_drop", default=0.0, type=float, help="State probability of dropout")
parser.add_argument("--n_run", default=1, type=int, help="Number of the experiment run")
parser.add_argument("--subclass", default='L2-3-IT', type=str, help="Cell subclass, e.g. Sst")
parser.add_argument("--exclude_type", default=[''], type=str, help="Types to exclude, e.g., 'L2/3 IT_2'")
parser.add_argument("--exclude_donors", default=[''], type=str, help="Donors to exclude")
parser.add_argument("--hard", default=False, type=bool, help="Enable hard encoding")
parser.add_argument("--device", default=0, type=int, help="GPU device, use None for CPU")
parser.add_argument("--saving_folder", default='./results/cpl_mixVAE/', type=str, help="Folder to save results")


def main(data_path, n_gene, n_categories, n_arm, state_dim, latent_dim, fc_dim, n_epoch, n_epoch_p, min_con, max_pron_it, batch_size,
         p_drop, s_drop, lr, temp, n_run, saving_folder, device, hard, tau, aug_path, subclass, exclude_type, exclude_donors, variational, beta):

    # Get the current working directory
    path = os.getcwd()

    # Load the data
    print(f'Loading AD {subclass} data ... ')
    with open(os.path.join(path, data_path, f'AD_MTG_{subclass}_nGene_{n_gene}_nDonor_84.p'), "rb") as f:
        data = pickle.load(f)
    print('Data is loaded')

    # Exclude specific cell types if specified
    if len(exclude_type) > 0:
        subclass_ind = np.array([i for i in range(len(data['supertype_scANVI'])) if data['supertype_scANVI'][i] not in exclude_type])
        print(np.unique(np.array(data['supertype_scANVI'])[subclass_ind]))
        ref_len = len(data['supertype_scANVI'])
        all_key = list(data.keys())
        for k in all_key:
            if len(data[k]) >= ref_len:
                if k == 'log1p':
                    data[k] = np.array(data[k])[subclass_ind, :]
                else:
                    data[k] = np.array(data[k])[subclass_ind]

    # Exclude specific donors if specified
    if len(exclude_donors) > 0:
        subcdata_ind = np.array([i for i in range(len(data['external_donor_name'])) if data['external_donor_name'][i] not in exclude_donors])
        print(np.unique(np.array(data['external_donor_name'])[subcdata_ind]))
        ref_len = len(data['supertype_scANVI'])
        all_key = list(data.keys())
        for k in all_key:
            if len(data[k]) >= ref_len:
                if k == 'log1p':
                    data[k] = np.array(data[k])[subcdata_ind, :]
                else:
                    data[k] = np.array(data[k])[subcdata_ind]

    print(data['log1p'].shape)

    # Create a folder name based on the experiment parameters
    folder_name = f'{subclass}_exc3Don_run_{n_run}_K_{n_categories}_Sdim_{state_dim}_ngene_{len(data["gene_id"])}_fcDim_{fc_dim}_' + \
                  f'latDim_{latent_dim}_lr_{lr}_pDrop_{p_drop}_n_arm_{n_arm}_tau_{tau}_bsize_{batch_size}_nepoch_{n_epoch}_nepochP_{n_epoch_p}'

    # Create directories for saving results
    os.makedirs(os.path.join(saving_folder, folder_name), exist_ok=True)
    saving_folder = os.path.join(saving_folder, folder_name)
    os.makedirs(os.path.join(saving_folder, 'model'), exist_ok=True)

    # Define the path to the augmenter file
    aug_file = os.path.join(aug_path, f'model_{subclass}_zdim_2_D_10_ngene_{n_gene}')

    # Initialize the coupled mixVAE (MMIDAS) model
    cpl_mixVAE = train_cplmixVAE(saving_folder=saving_folder, device=device, aug_file=aug_file)

    # Get data loaders
    alldata_loader, train_loader, val_set_torch, validation_loader, test_set_torch, test_loader = cpl_mixVAE.getdata(
        dataset=data['log1p'], label=data['supertype_scANVI'], batch_size=batch_size)

    # Initialize the model with the given parameters
    cpl_mixVAE.init_model(n_categories=n_categories, state_dim=state_dim, input_dim=data['log1p'].shape[1], fc_dim=fc_dim, lowD_dim=latent_dim,
                          x_drop=p_drop, s_drop=s_drop, lr=lr, n_arm=n_arm, temp=temp, hard=hard, tau=tau, variational=variational, beta=beta)

    # Run the model
    cpl_mixVAE.run(train_loader, test_loader, val_set_torch, alldata_loader, n_epoch, n_epoch_p, min_con, max_pron_it)


# Run the main function if this script is executed directly
if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))

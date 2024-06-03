import argparse
import os
import numpy as np
from utils.training import train_cplmixVAE
from utils.helpers import load_config, load_patchseq
import matplotlib.pyplot as plt

# Setup argument parser for command line arguments
parser = argparse.ArgumentParser()

# Define arguments with default values and help descriptions
parser.add_argument("--n_categories", default=100, type=int, help="Number of cell types")
parser.add_argument("--state_dim_T", default=2, type=int, help="State variable dimension for T data")
parser.add_argument("--state_dim_E", default=3, type=int, help="State variable dimension for E data")
parser.add_argument("--n_arm_T", default=2, type=int, help="Number of mixVAE arms for T modality")
parser.add_argument("--n_arm_E", default=2, type=int, help="Number of mixVAE arms for E modality")
parser.add_argument("--n_modal", default=2, type=int, help="Number of data modalities")
parser.add_argument("--temp", default=1.0, type=float, help="Gumbel-softmax temperature")
parser.add_argument("--tau", default=.01, type=float, help="Softmax temperature")
parser.add_argument("--beta", default=1, type=float, help="Beta factor for regularization")
parser.add_argument("--lam_T", default=1, type=int, help="Coupling factor between T arms")
parser.add_argument("--lam_E", default=1, type=int, help="Coupling factor between E arms")
parser.add_argument("--lam_TE", default=1, type=int, help="Coupling factor between TE arms")
parser.add_argument("--latent_dim_T", default=30, type=int, help="Latent dimension of T")
parser.add_argument("--latent_dim_E", default=30, type=int, help="Latent dimension of E")
parser.add_argument("--n_epoch", default=10000, type=int, help="Number of epochs to train")
parser.add_argument("--n_epoch_p", default=10000, type=int, help="Number of epochs to train pruning algorithm")
parser.add_argument("--max_prun_it", default=60, type=int, help="Maximum number of pruning iterations")
parser.add_argument("--min_con", default=.95, type=float, help="Minimum consensus for pruning")
parser.add_argument("--min_density", default=5, type=int, help="Minimum number of samples in a class")
parser.add_argument("--aug_file", default='augModel_T', type=str, help="Path of the data augmenter")
parser.add_argument("--n_aug_smp", default=0, type=int, help="Number of augmented samples")
parser.add_argument("--fc_dim_T", default=100, type=int, help="Number of nodes at the hidden layers for T data")
parser.add_argument("--fc_dim_E", default=100, type=int, help="Number of nodes at the hidden layers for E data")
parser.add_argument("--batch_size", default=1000, type=int, help="Batch size")
parser.add_argument("--variational", default='true', type=str, help="Enable variational mode")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
parser.add_argument("--p_drop_T", default=0.25, type=float, help="Input probability of dropout for T data")
parser.add_argument("--p_drop_E", default=0.0, type=float, help="Input probability of dropout for E data")
parser.add_argument("--noise_std", default=0.05, type=float, help="Additive noise standard deviation for E data")
parser.add_argument("--s_drop_T", default=0.0, type=float, help="State probability of dropout for T")
parser.add_argument("--s_drop_E", default=0.0, type=float, help="State probability of dropout for E")
parser.add_argument("--n_run", default=1, type=int, help="Number of the experiment run")
parser.add_argument("--hard", default='false', type=str, help="Enable hard encoding")
parser.add_argument("--device", default=0, type=int, help="GPU device, use None for CPU")
parser.add_argument("--saving_folder", default='results', type=str, help="Path for saving results")


def main(n_categories, n_arm_T, n_arm_E, n_modal, state_dim_T, state_dim_E, latent_dim_T, latent_dim_E, fc_dim_T, fc_dim_E, n_epoch, n_epoch_p, min_con, min_density,
         batch_size, p_drop_T, p_drop_E, s_drop_T, s_drop_E, noise_std, lr, temp, n_run, saving_folder, device, hard, tau, aug_file, variational, max_prun_it, beta,
         lam_T, lam_E, lam_TE, n_aug_smp):

    # Load configuration settings from the config file
    paths = load_config('config.toml')
    saving_folder = paths['package_dir'] / saving_folder

    # Create a folder name based on the experiment parameters
    folder_name = f'run_{n_run}_proc_K_{n_categories}_SdimT_{state_dim_T}_SdimE_{state_dim_E}_' + f'lr_{lr}_n_armT_{n_arm_T}_n_armE_{n_arm_E}_tau_{tau}_nbatch_{batch_size}_nepoch_{n_epoch}_nepochP_{n_epoch_p}'

    # Update augmenter file path if provided
    if aug_file:
        aug_file = 'augmenter/' + aug_file
        aug_file = saving_folder / aug_file

    # Create directories for saving results
    saving_folder = saving_folder / folder_name
    os.makedirs(saving_folder, exist_ok=True)
    os.makedirs(saving_folder / 'model', exist_ok=True)
    saving_folder = str(saving_folder)

    # Create dictionaries for model parameters
    state_dim = {'T': state_dim_T, 'E': state_dim_E}
    fc_dim = {'T': fc_dim_T, 'E': fc_dim_E}
    latent_dim = {'T': latent_dim_T, 'E': latent_dim_E}
    p_drop = {'T': p_drop_T, 'E': p_drop_E}
    s_drop = {'T': s_drop_T, 'E': s_drop_E}
    n_arm = {'T': n_arm_T, 'E': n_arm_E}
    lam = {'T': lam_T, 'E': lam_E, 'TE': lam_TE}

    # Load Patch-seq data and exclude specified cell types
    rmv_cell = ['Sst Crh 4930553C11Rik           ', 'Sst Myh8 Etv1                   ']
    D = load_patchseq(path=paths['data'], exclude_type=rmv_cell, min_num=5)
    data_T = D['XT']
    data_E = D['XE']
    mask = {'ET': D['crossModal_id'], 'TE': D['crossModal_id'], 'E': ~D['isnan_E'], 'T': ~D['isnan_T']}

    input_dim = {'T': data_T.shape[1], 'E': data_E.shape[1]}

    # Determine if the model should use variational mode or not
    state_det = variational != 'true'

    # Determine if the model should use hard encoding or not
    b_hard = hard == 'true'

    # Initialize the coupled mixVAE (MMIDAS) model
    cpl_mixVAE = train_cplmixVAE(saving_folder=saving_folder, device=device, aug_file=aug_file)

    # Get data loaders
    alldata_loader, train_loader, validation_loader, test_loader = cpl_mixVAE.getdata(
        dataset_T=data_T, dataset_E=data_E, label=D['cluster_label'], batch_size=batch_size, n_aug_smp=n_aug_smp)

    # Initialize the model with the given parameters
    cpl_mixVAE.init_model(n_categories=n_categories, state_dim=state_dim, input_dim=input_dim, fc_dim=fc_dim,
                          lowD_dim=latent_dim, x_drop=p_drop, s_drop=s_drop, noise_std=noise_std, state_det=state_det,
                          beta=beta, lr=lr, n_arm=n_arm, n_modal=n_modal, temp=temp, hard=b_hard, tau=tau, lam=lam)

    # Run the model
    cpl_mixVAE.run(train_loader, validation_loader, test_loader, alldata_loader, mask, n_epoch, n_epoch_p, min_con, max_prun_it, min_density)

    # Evaluate the model and save scatter plots of the state means
    outcome_dict = cpl_mixVAE.eval_model(data_T, data_E, mask)
    plt.figure()
    tmp = np.concatenate(outcome_dict['state_mu']['T'][0])
    plt.scatter(tmp[:, 0], tmp[:, 1])
    plt.savefig(saving_folder + '/state_mu_T.png', dpi=600)
    plt.figure()
    tmp = np.concatenate(outcome_dict['state_mu']['E'][0])
    if state_dim_E > 1:
        plt.scatter(tmp[:, 0], tmp[:, 1])
        plt.savefig(saving_folder + '/state_mu_E.png', dpi=600)
    plt.close('all')


# Run the main function if this script is executed directly
if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))

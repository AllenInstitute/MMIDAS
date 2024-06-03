import argparse
import os
import numpy as np
from mmidas.cpl_mixvae import cpl_mixVAE
from mmidas.utils.tools import get_paths
from mmidas.utils.dataloader import load_data, get_loaders

# Setup argument parser for command line arguments
parser = argparse.ArgumentParser()

# Define command line arguments
parser.add_argument("--n_categories", default=120, type=int, help="(maximum) number of cell types")
parser.add_argument("--state_dim", default=2, type=int, help="state variable dimension")
parser.add_argument("--n_arm", default=2, type=int, help="number of mixVAE arms for each modality")
parser.add_argument("--temp", default=1, type=float, help="gumbel-softmax temperature")
parser.add_argument("--tau", default=.005, type=float, help="softmax temperature")
parser.add_argument("--beta", default=.01, type=float, help="KL regularization parameter")
parser.add_argument("--lam", default=1, type=float, help="coupling factor")
parser.add_argument("--lam_pc", default=1, type=float, help="coupling factor for ref arm")
parser.add_argument("--latent_dim", default=10, type=int, help="latent dimension")
parser.add_argument("--n_epoch", default=10000, type=int, help="Number of epochs to train")
parser.add_argument("--n_epoch_p", default=10000, type=int, help="Number of epochs to train pruning algorithm")
parser.add_argument("--min_con", default=.99, type=float, help="minimum consensus")
parser.add_argument("--max_prun_it", default=50, type=int, help="maximum number of pruning iterations")
parser.add_argument("--ref_pc", default=False, type=bool, help="use a reference prior component")
parser.add_argument("--fc_dim", default=100, type=int, help="number of nodes at the hidden layers")
parser.add_argument("--batch_size", default=5000, type=int, help="batch size")
parser.add_argument("--variational", default=True, type=bool, help="enable variational mode")
parser.add_argument("--augmentation", default=False, type=bool, help="enable VAE-GAN augmentation")
parser.add_argument("--lr", default=.001, type=float, help="learning rate")
parser.add_argument("--p_drop", default=0.5, type=float, help="input probability of dropout")
parser.add_argument("--s_drop", default=0.2, type=float, help="state probability of dropout")
parser.add_argument("--pretrained_model", default=False, type=bool, help="use pretrained model")
parser.add_argument("--n_pr", default=0, type=int, help="number of pruned categories in case of using a pretrained model")
parser.add_argument("--loss_mode", default='MSE', type=str, help="loss mode, MSE or ZINB")
parser.add_argument("--n_run", default=1, type=int, help="number of the experiment")
parser.add_argument("--hard", default=False, type=bool, help="hard encoding")
parser.add_argument("--device", default=None, type=int, help="computing device, either 'cpu' or 'cuda'.")


# Main function
def main(n_categories, n_arm, state_dim, latent_dim, fc_dim, n_epoch, n_epoch_p, min_con, max_prun_it, batch_size, lam, lam_pc, loss_mode,
         p_drop, s_drop, lr, temp, n_run, device, hard, tau, variational, ref_pc, augmentation, pretrained_model, n_pr, beta):

    # Load configuration paths
    toml_file = 'pyproject.toml'
    sub_file = 'smartseq_files'
    config = get_paths(toml_file=toml_file, sub_file=sub_file)
    data_path = config['paths']['main_dir'] / config['paths']['data_path']
    data_file = data_path / config[sub_file]['anndata_file']

    # Define folder name for saving results
    folder_name = f'run_{n_run}_K_{n_categories}_Sdim_{state_dim}_aug_{augmentation}_lr_{lr}_n_arm_{n_arm}_nbatch_{batch_size}' + \
                  f'_train.ipynb_nepoch_{n_epoch}_nepochP_{n_epoch_p}'
    saving_folder = config['paths']['main_dir'] / config['paths']['saving_path']
    saving_folder = saving_folder / folder_name
    os.makedirs(saving_folder, exist_ok=True)
    os.makedirs(saving_folder / 'model', exist_ok=True)
    saving_folder = str(saving_folder)

    # Determine augmentation file path
    if augmentation:
        aug_file = config['paths']['main_dir'] / config[sub_file]['aug_model']
    else:
        aug_file = ''

    # Determine pretrained model file path
    if pretrained_model:
        trained_model = config['paths']['main_dir'] / config[sub_file]['trained_model']
    else:
        trained_model = ''

    # Load data
    data = load_data(datafile=data_file)
    trainloader, testloader, _ = get_loaders(dataset=data['log1p'], batch_size=batch_size)

    # Initialize the coupled mixVAE (MMIDAS) model
    cplMixVAE = cpl_mixVAE(saving_folder=saving_folder, device=device)

    # Initialize the model with specified parameters
    cplMixVAE.init_model(n_categories=n_categories,
                         state_dim=state_dim,
                         input_dim=data['log1p'].shape[1],
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
                         ref_prior=ref_pc,
                         variational=variational,
                         trained_model=trained_model,
                         n_pr=n_pr,
                         mode=loss_mode)

    # Train and save the model
    model_file = cplMixVAE.train(train_loader=trainloader,
                                 test_loader=testloader,
                                 n_epoch=n_epoch,
                                 n_epoch_p=n_epoch_p,
                                 c_onehot=data['c_onehot'],
                                 c_p=data['c_p'],
                                 min_con=min_con,
                                 max_prun_it=max_prun_it)


# Run the main function when the script is executed
if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))

import argparse
import os
import torch

from mmidas.cpl_mixvae import cpl_mixVAE
from mmidas.vaegan import vae_gan
from mmidas.utils.config_tools import get_paths
from mmidas.utils.data_tools import load_data, get_loaders


parser = argparse.ArgumentParser()
parser.add_argument("--n_categories", default=15, type=int, help="number of cell types")
parser.add_argument("--state_dim", default=3, type=int, help="state variable dimension")
parser.add_argument("--n_arm", default=2, type=int,  help="number of mixVAE arms for each modalities")
parser.add_argument("--temp",  default=1, type=float, help="gumbel-softmax temperature")
parser.add_argument("--tau",  default=.1, type=float, help="softmax temperature")
parser.add_argument("--beta",  default=1, type=float, help="KL regularization parameter")
parser.add_argument("--lam",  default=1, type=float, help="coupling factor")
parser.add_argument("--lam_pc",  default=1, type=float, help="coupling factor for ref arm")
parser.add_argument("--ref_pc", default=False, action="store_true", help="using reference arm")
parser.add_argument("--latent_dim", default=10, type=int, help="latent dimension")
parser.add_argument("--n_epoch", default=10000, type=int, help="Number of epochs to train")
parser.add_argument("--n_epoch_p", default=10000, type=int, help="Number of epochs to train pruning algorithm")
parser.add_argument("--min_con", default=.99, type=float, help="minimum consensus")
parser.add_argument("--max_prun_it", default=14, type=int, help="maximum number of pruning iterations")
parser.add_argument("--n_aug_smp", default=0, type=int, help="number of augmented samples")
parser.add_argument("--fc_dim", default=100, type=int, help="number of nodes at the hidden layers")
parser.add_argument("--batch_size", default=256, type=int, help="batch size")
parser.add_argument("--variational", default=True, help="enable variational mode")
parser.add_argument("--augmentation", default=False, action="store_true", help="enable VAE-GAN augmentation")
parser.add_argument("--lr", default=.001, type=float, help="learning rate")
parser.add_argument("--n_gene", default=0., type=int, help="number of genes")
parser.add_argument("--p_drop", default=0.25, type=float, help="input probability of dropout")
parser.add_argument("--s_drop", default=0.0, type=float, help="state probability of dropout")
parser.add_argument("--n_run", default=2, type=int, help="number of the experiment")
parser.add_argument("--hard", default=False, action="store_true", help="hard encoding")
parser.add_argument("--pre_trained_model", default='', type=str, help="pre-trained model")
parser.add_argument("--n_prun_c", default=0, type=int, help="number of prunned categories")
parser.add_argument("--training_mode", default='MSE', type=str, help="mode of the reconstruction loss: MSE or ZINB")
parser.add_argument("--seed", default=0, type=int, help="random seed")
parser.add_argument("--cuda", default=False, action="store_true", help="enable cuda (gpu device)")
parser.add_argument("--toml_file", default='pyproject.toml', type=str, help="the project toml file")


def main(
        n_categories, 
        n_arm, 
        state_dim, 
        latent_dim, 
        fc_dim, 
        n_epoch, 
        n_epoch_p, 
        min_con,
        max_prun_it, 
        batch_size, 
        n_aug_smp,
        p_drop, 
        s_drop, 
        lr, 
        temp, 
        n_run, 
        cuda, 
        hard, 
        tau, 
        variational, 
        ref_pc, 
        augmentation, 
        n_gene, 
        lam, 
        lam_pc, 
        beta, 
        pre_trained_model,
        n_prun_c,
        training_mode,
        toml_file,
        seed,
        ):

    config = get_paths(toml_file=toml_file)
    data_file_1 = config['paths']['main_dir'] / config['paths']['data_path'] / config['data']['anndata_file_1']
    data_file_2 = config['paths']['main_dir'] / config['paths']['data_path'] / config['data']['anndata_file_2']
    data_file_3 = config['paths']['main_dir'] / config['paths']['data_path'] / config['data']['anndata_file_3']
    
    gene_file = config['paths']['main_dir'] / config['paths']['data_path'] / config['data']['hvg_file_2']
    
    folder_name = f'run_{n_run}_Cdim_{n_categories}_Sdim_{state_dim}_Zdim_{latent_dim}_pdrop_{p_drop}_fcdim_{fc_dim}_aug_{augmentation}' + \
                  f'_lr_{lr}_narm_{n_arm}_tau_{tau}_nbatch_{batch_size}_nepoch_{n_epoch}_nepochP_{n_epoch_p}_dataset_all'
    
    saving_folder = config['paths']['main_dir'] / config['paths']['saving_path']
    saving_folder = saving_folder / folder_name
    os.makedirs(saving_folder, exist_ok=True)
    os.makedirs(saving_folder / 'model', exist_ok=True)
    saving_folder = str(saving_folder)
    
    if cuda:
        free_gpus = []
        for i in range(torch.cuda.device_count()):
            if torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i) > 0:
                free_gpus.append(i)
        if free_gpus:
            device = torch.device(f"cuda:{free_gpus[0]}")
        else:
            raise RuntimeError("No free GPU devices available.")
    else:
        device = torch.device("cpu")

    if augmentation:
        aug_path = config['paths']['main_dir'] / config['paths']['models']
        aug_file = aug_path / config['models']['augmenter_2']
        aug_vaegan = vae_gan(saving_folder=aug_path, device=device)
        aug_vaegan.load_model(aug_file)
        augmenter = aug_vaegan.netA
    else:
        augmenter = []

    data_files = [data_file_1, data_file_2, data_file_3]
    data = load_data(file=data_files, gene_file=gene_file, n_gene=n_gene) 

    mixvae = cpl_mixVAE(saving_folder=saving_folder, augmenter=augmenter, device=device)
    
    _, train_loader, test_loader, _, _, _ = get_loaders(
                                                        x=data['log1p'],
                                                        batch_size=batch_size, 
                                                        n_aug_smp=n_aug_smp, 
                                                        additional_val=False,
                                                        seed=seed,
                                                        )

    mixvae.init_model(
                    n_categories=n_categories,
                    state_dim=state_dim,
                    input_dim=data['log1p'].shape[1],
                    fc_dim=fc_dim,
                    lowD_dim=latent_dim,
                    x_drop=p_drop,
                    s_drop=s_drop,
                    n_arm=n_arm,
                    temp=temp,
                    hard=hard,
                    variational=variational,
                    tau=tau,
                    lam=lam,
                    lam_pc=lam_pc,
                    beta=beta,
                    ref_prior=ref_pc,
                    mode=training_mode,
                    trained_model=pre_trained_model,
                    n_pr=n_prun_c,
                    )
    

    mixvae.train(
                train_loader=train_loader,
                test_loader=test_loader,
                n_epoch=n_epoch,
                n_epoch_p=n_epoch_p,
                lr=lr,
                min_con=min_con,
                max_prun_it=max_prun_it,
                )
    

if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))

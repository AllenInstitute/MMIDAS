import argparse
import os
import torch
import pandas as pd
import pdb

from mmidas.vaegan import vae_gan
from mmidas.cpl_mixvae import cpl_mixVAE
from mmidas.utils.config_tools import get_paths
from mmidas.utils.augmentation import get_loader as aug_loader, freeze
from mmidas.utils.data_tools import load_data, get_loaders, Dbh_Retro_loaders


parser = argparse.ArgumentParser()
parser.add_argument("--z_dim", default=10, type=int, help="latent dimension")
parser.add_argument("--noise_dim", default=50, type=int, help="noise dimension")
parser.add_argument("--alpha", default=.2, type=float,  help="triple loss parameter")
parser.add_argument("--n_gene", default=0, type=int, help="number of genes")
parser.add_argument("--n_epoch_aug", default=20000, type=int, help="Number of epochs to train")
parser.add_argument("--fc_dim_aug", default=500, type=int, help="number of nodes at the hidden layers")
parser.add_argument("--batch_size", default=128, type=int, help="batch size")
parser.add_argument("--affine", default=False, action="store_true", help="affine transformation in the batch normalization")
parser.add_argument("--momentum",  default=0.01, type=float, help="momentum for batch normalization")
parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
parser.add_argument("--p_drop", default=0.25, type=float, help="input probability of dropout")
parser.add_argument("--n_categories", default=10, type=int, help="number of cell types")
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
parser.add_argument("--min_con", default=1., type=float, help="minimum consensus")
parser.add_argument("--max_prun_it", default=9, type=int, help="maximum number of pruning iterations")
parser.add_argument("--n_aug_smp", default=0, type=int, help="number of augmented samples")
parser.add_argument("--fc_dim", default=100, type=int, help="number of nodes at the hidden layers")
parser.add_argument("--variational", default=True, help="enable variational mode")
parser.add_argument("--augmentation", default=False, action="store_true", help="enable VAE-GAN augmentation")
parser.add_argument("--s_drop", default=0.0, type=float, help="state probability of dropout")
parser.add_argument("--n_run", default=7, type=int, help="number of the experiment")
parser.add_argument("--hard", default=False, action="store_true", help="hard encoding")
parser.add_argument("--pre_trained_model", default='', type=str, help="pre-trained model")
parser.add_argument("--n_prun_c", default=0, type=int, help="number of prunned categories")
parser.add_argument("--training_mode", default='MSE', type=str, help="mode of the reconstruction loss: MSE or ZINB")
parser.add_argument("--seed", default=0, type=int, help="random seed")
parser.add_argument("--cuda", default=False, action="store_true", help="enable cuda (gpu device)")
parser.add_argument("--toml_file", default='pyproject.toml', type=str, help="the project toml file")
parser.add_argument("--data_file", default='Retroseq_updated', type=str, help="the data file")
parser.add_argument("--platform", type=str, help="if you want to run the code on a specific platform, please specify it here ('Dbh' or 'Retroseq')")


def main(args):

    config = get_paths(toml_file=args.toml_file)
    data_file = config['paths']['main_dir'] / config['paths']['data_path'] / config['data'][f'{args.data_file}_file']
    saving_folder = config['paths']['main_dir'] / config['paths']['saving_path']
    data = load_data(file=data_file, n_gene=args.n_gene) 

    # if args.platform:
    #     indx = Dbh_Retroseq_data['dataset'] == args.platform
    #     Dbh_Retroseq_data['log1p'] = Dbh_Retroseq_data['log1p'][indx]
    #     for key in Dbh_Retroseq_data:
    #         if key != 'log1p' and key != 'gene_id':
    #             Dbh_Retroseq_data[key] = Dbh_Retroseq_data[key][indx]


    n_gene = data['log1p'].shape[1]
    folder_name = f'run_{args.n_run}_Cdim_{args.n_categories}_Sdim_{args.state_dim}_Zdim_{args.latent_dim}_pdrop_{args.p_drop}_fcdim_{args.fc_dim}_aug_{args.augmentation}' + \
                  f'_naug_{args.n_aug_smp}_lr_{args.lr}_narm_{args.n_arm}_tau_{args.tau}_nbatch_{args.batch_size}_nepoch_{args.n_epoch}_nepochP_{args.n_epoch_p}_{args.data_file}'
    
    
    saving_folder = saving_folder / folder_name
    os.makedirs(saving_folder, exist_ok=True)
    os.makedirs(saving_folder / 'model', exist_ok=True)
    with open(saving_folder / 'parameter.text', "w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
    
    saving_folder = str(saving_folder)
    
        
    if args.cuda:
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
    
    
    if args.augmentation:
        aug_path = config['paths']['main_dir'] / config['paths']['models']
        os.makedirs(aug_path, exist_ok=True)
        aug_vaegan = vae_gan(saving_folder=aug_path, device=device)
        aug_vaegan.init_model(
                            input_dim=n_gene,
                            z_dim=args.z_dim, 
                            noise_dim=args.noise_dim, 
                            fc_dim=args.fc_dim_aug, 
                            x_drop=args.p_drop, 
                            affine=args.affine, 
                            momentum=args.momentum,
                            )
        
        if str(config['models'][f'augmenter_{args.platform}']) == '.':
            data_loader = aug_loader(x=data['log1p'], batch_size=args.batch_size, training=True)
            aug_model = aug_vaegan.train(
                                        dataloader=data_loader, 
                                        n_epoch=args.n_epoch_aug, 
                                        lr=args.lr, 
                                        alpha=args.alpha, 
                                        lam = [1, 0.5, .1, .5], 
                                        tag=args.data_file,
                                        )
        else:
            aug_model = aug_path / config['models'][f'augmenter_{args.platform}']
        
        aug_file = aug_path / aug_model
        aug_vaegan.load_model(aug_file)
        augmenter = aug_vaegan.netA
        freeze(augmenter)
        
    else:
        augmenter = []

    if args.platform:
        _, train_loader, test_loader, _, _, _ = get_loaders(
                                                            x=data['log1p'],
                                                            batch_size=args.batch_size, 
                                                            n_aug_smp=args.n_aug_smp, 
                                                            netA=augmenter.to('cpu'),
                                                            additional_val=False,
                                                            seed=args.seed,
                                                            )
    else:
        x_Dbh = data['log1p'][data['dataset'] == 'Dbh']
        x_Retroseq = data['log1p'][data['dataset'] == 'Retroseq'] 
        label = data['injection_target'][data['dataset'] == 'Retroseq']
        _, train_loader, test_loader, _, _, _ = Dbh_Retro_loaders(
                                                                    x_Dbh=x_Dbh,
                                                                    x_Retro=x_Retroseq,
                                                                    batch_size=args.batch_size, 
                                                                    label=label,
                                                                    n_aug_smp=args.n_aug_smp, 
                                                                    netA=augmenter.to('cpu'),
                                                                    additional_val=False,
                                                                    seed=args.seed,
                                                                    )
    del data
    # print(train_loader.dataset.tensors[0])
    # print(test_loader.dataset.tensors[0])
    
    mixvae = cpl_mixVAE(saving_folder=saving_folder, augmenter=augmenter, device=device)
    mixvae.init_model(
                    n_categories=args.n_categories,
                    state_dim=args.state_dim,
                    input_dim=n_gene,
                    fc_dim=args.fc_dim,
                    lowD_dim=args.latent_dim,
                    x_drop=args.p_drop,
                    s_drop=args.s_drop,
                    n_arm=args.n_arm,
                    temp=args.temp,
                    hard=args.hard,
                    variational=args.variational,
                    tau=args.tau,
                    lam=args.lam,
                    lam_pc=args.lam_pc,
                    beta=args.beta,
                    ref_prior=args.ref_pc,
                    mode=args.training_mode,
                    trained_model=args.pre_trained_model,
                    n_pr=args.n_prun_c,
                    )
    
    # pdb.set_trace()
    mixvae.train(
                train_loader=train_loader,
                test_loader=test_loader,
                n_epoch=args.n_epoch,
                n_epoch_p=args.n_epoch_p,
                lr=args.lr,
                min_con=args.min_con,
                max_prun_it=args.max_prun_it,
                )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

    
    
    
    
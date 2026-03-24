import argparse
import os
import torch

from mmidas.vaegan import vae_gan
from mmidas.utils.config_tools import get_paths
from mmidas.utils.data_tools import load_data
from mmidas.utils.augmentation import get_loader


parser = argparse.ArgumentParser()
parser.add_argument("--z_dim", default=10, type=int, help="latent dimension")
parser.add_argument("--noise_dim", default=50, type=int, help="noise dimension")
parser.add_argument("--alpha", default=.2, type=float,  help="triple loss parameter")
parser.add_argument("--n_gene", default=0, type=int, help="number of genes")
parser.add_argument("--n_epoch", default=10, type=int, help="Number of epochs to train")
parser.add_argument("--fc_dim", default=500, type=int, help="number of nodes at the hidden layers")
parser.add_argument("--batch_size", default=512, type=int, help="batch size")
parser.add_argument("--affine", default=False, action="store_true", help="affine transformation in the batch normalization")
parser.add_argument("--momentum",  default=0.01, type=float, help="momentum for batch normalization")
parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
parser.add_argument("--p_drop", default=0.25, type=float, help="input probability of dropout")
parser.add_argument("--cuda", default=False, action="store_true", help="enable cuda (gpu device)")
parser.add_argument("--toml_file", default='pyproject.toml', type=str, help="the project toml file")


def main(z_dim, noise_dim, alpha, n_gene, n_epoch, fc_dim, batch_size, affine, lr, p_drop, momentum, cuda, toml_file):

    config = get_paths(toml_file=toml_file)
    data_file = config['paths']['main_dir'] / config['paths']['data_path'] / config['data']['anndata_file_1']
    gene_file = config['paths']['main_dir'] / config['paths']['data_path'] / config['data']['hvg_file_2']
    print(f"gene_file: {gene_file}")
    
    data = load_data(file=data_file, gene_file=gene_file, n_gene=n_gene) 
    
    saving_folder = config['paths']['main_dir'] / config['paths']['models']
    os.makedirs(saving_folder, exist_ok=True)
    
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
        

    augmenter = vae_gan(saving_folder=saving_folder, device=device)
    
    augmenter.init_model(
                        input_dim=data['log1p'].shape[1], 
                        z_dim=z_dim, 
                        noise_dim=noise_dim, 
                        fc_dim=fc_dim, 
                        x_drop=p_drop, 
                        affine=affine, 
                        momentum=momentum,
                        )
    
    data_loader = get_loader(x=data['log1p'], batch_size=batch_size, training=True)
    augmenter.train(
                    dataloader=data_loader, 
                    n_epoch=n_epoch, 
                    lr=lr, 
                    alpha=alpha, 
                    lam = [1, 0.5, .1, .5], 
                    )
    

if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))

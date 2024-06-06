from utils.augmentation.train import *
from utils.config import load_config

# Use GPU if available
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
print(device, " will be used.\n")

# Load the configuration file
paths = load_config(config_file='config.toml')
# Define the path to the data file and the saving path
data_file = '/xxx/xxx.h5ad'
saving_path = '/xxx/'

# Dictionary of the training parameters for CTX-HIP datatset
parameters = {'batch_size': 1000,  # batch size
            'num_epochs': 1000,  # number of epochs
            'learning_rate': 1e-3, # learning rate
            'alpha': 0.2,  # triplet loss hyperparameter
            'num_z': 10, # latent space dimension
            'num_n': 50, # noise dimension
            'lambda': [1, 0.5, 0.1, 0.5], # weights of the augmenter loss
            'dataset_file': data_file,
            'feature': 'log1p',
            'subclass': '',
            'n_zim': 2,
            'n_smp': 20, # number of augmented samples
            'initial_w': False, # initial weights
            'affine': False,
            'n_run': 1,
            'save': 'True', # saving flag
            'file_name':  saving_path + 'augmenter_model',
            'saving_path': saving_path
            }

train_udagan(parameters, device)

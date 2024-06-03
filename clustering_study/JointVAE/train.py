import torch, pickle
from jointvae.newModel import VAE
from jointvae.training import Trainer
from utils.dataloaders import gene_dataloader, load_10x_data
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score

tau_c = [.005]
lowD_c = [10]
batch_c = [5000]
state_dim_c = [3]
temp = [.1]
date = 20240119

lr = 5e-4
epochs = 2000
n_features = 10000
data_path = '/xxx/CTX-HIP/mouse/brain_map_10x/'

# set the cell types data file
subclass = 'Glutamatergic' #'GABAergic'  
measure='log1p'
data_file = f'{subclass}_isoCTX_nGene_10000.h5ad'
dataset = load_10x_data(data_path + data_file, measure=measure)
n_gene = dataset[measure].shape[1]

# Check for cuda
use_cuda = torch.cuda.is_available()
label_original = []
label_estimated = []
adj_rand_indx = []
adj_MI_indx = []
accuracy = []
rec_loss = []
num_category = []

for select in range(1):

    batch_size = batch_c[select]
    class_label = np.array(list(dataset['cluster_label']))
    n_cat = len(np.unique(class_label))
    # Load data
    data_loader, _, _, allData_loader = gene_dataloader(dataset, n_features, batch_size)

    # Define latent spec and model
    latent_spec = {'cont': state_dim_c[select], 'disc': [n_cat]}
    model = VAE(input_dim=n_gene, lowD_dim=lowD_c[select], latent_spec=latent_spec,
                use_cuda=use_cuda, temperature=temp[select])
    if use_cuda:
        model.cuda()

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Define trainer
    trainer = Trainer(model, optimizer,
                      cont_capacity=[0.0, 10 * np.log(state_dim_c[select]),
                                     10 * epochs,
                                     100],
                      disc_capacity=[0.0, 2 * np.log(n_cat), 10 * epochs,
                                     100],
                      use_cuda=use_cuda)

    # Train model for 100 epochs
    trainer.train(data_loader, epochs)

    # Save trained model
    torch.save(trainer.model.state_dict(), f'./model_{subclass}_10x_K_{n_cat}_{date}.pt')

    num_smp = len(allData_loader.dataset)
    # arr_size = int(len(allData_loader) / batch_size) + 1
    class_label = []
    category_vs_class = np.zeros((n_cat, n_cat))
    predicted_label = []
    latent_x = []
    state_mu = np.zeros((num_smp, 10))
    # Extract a batch of data
    loss = []

    for i, (data, labels) in enumerate(allData_loader):
        d_len = len(data)
        data = Variable(data).cuda()
        recon_batch, hidden, latent_dist = model(data)
        recon_loss = F.mse_loss(recon_batch, data, reduction='mean')

        loss.append(recon_loss.cpu().detach().numpy())
        _, encodings = model.encode(data)

        l = [int(lab) for lab in labels.numpy()]
        # lowD_rep[arm, i * state[arm].size(0):(i + 1) * state[
        #     arm].size(0), :] = data_low[arm].cpu().detach().numpy()
        label = dataset['cluster_id'][l]

        class_label.append(label)

        z_smp = encodings['disc'][0].cpu().detach().numpy()

        label_predict = []
        for d in range(len(label)):
            z_cat = np.squeeze(z_smp[d, :])
            category_vs_class[int(label[d] - 1), np.argmax(z_cat)] += 1
            label_predict.append(np.argmax(z_cat) + 1)

        predicted_label.append(np.array(label_predict))
        latent_x.append(hidden.cpu().detach().numpy())

    class_label = np.concatenate(class_label)
    predicted_label = np.concatenate(predicted_label)
    unique_test_class_label = np.unique(class_label)
    unique_types = [dataset['cluster_label'][dataset['cluster_id'] == cl][0] for cl in unique_test_class_label]

    label_original.append(class_label)
    label_estimated.append(predicted_label)
    adj_rand_indx.append(adjusted_rand_score(class_label, predicted_label))
    adj_MI_indx.append(adjusted_mutual_info_score(class_label, predicted_label))
    rec_loss.append(np.mean(loss))

    num_category.append(n_cat)

print('Loss')
print(rec_loss)

dict = {}
f = open(f'./jointVAE_10x_{subclass}_{date}.p', "wb")
dict['recon_loss'] = rec_loss
dict['num_category'] = num_category
dict['ttype'] = class_label
dict['cluster'] = predicted_label
dict['lowD_x'] = np.concatenate(latent_x)
dict['sample_id'] = dataset['sample_name']
pickle.dump(dict, f)
f.close()


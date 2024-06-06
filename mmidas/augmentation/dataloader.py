import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def get_genes(gene_id, n_genes):
    """
    Get gene indices for the selected genes

    Args:
        gene_id (list): list of gene IDs
        n_genes (int): number of genes

    Returns:
        gene_index (np.array): gene indices
    """

    # Define the genes for each neuronal type, here, we use the genes from the Allen Brain Atlas
    gaba_ind_1, gaba_ind_2, glutam_ind = np.array([]), np.array([]), np.array([])
    glutam_genes = ['Slc30a3', 'Cux2', 'Rorb', 'Deptor', 'Scnn1a', 'Rspo1',
                    'Hsd11b1', 'Batf3', 'Oprk1', 'Osr1', 'Car3', 'Fam84b',
                    'Chrna6', 'Pvalb', 'Pappa2', 'Foxp2', 'Slc17a8', 'Trhr',
                    'Tshz2', 'Rapdegf3', 'Trh', 'Gpr139', 'Nxph4', 'Rprm',
                    'Crym', 'Nxph3', 'Nlgn1', 'C1ql2', 'C1ql3', 'Adgrl1', 'Nlgn3',
                    'Dag1', 'Cbln1', 'Lrrtm1']

    gaba_genes_1 = ['Lamp5', 'Ndnf', 'Krt73', 'Fam19a1', 'Pax6', 'Ntn1', 'Plch2',
                    'Lsp1', 'Lhx6', 'Nkx2.1', 'Vip', 'Sncg', 'Slc17a8', 'Nptx2',
                    'Gpr50', 'Itih5', 'Serpinf1', 'Igfbp6', 'Gpc3', 'Lmo1',
                    'Ptprt', 'Rspo4', 'Chat', 'Crispld2', 'Col15a1', 'Pde1a',
                    'Cbln2', 'Cbln4', 'C1ql1', 'Lrrtm3', 'Clstn3', 'Nlgn2',
                    'Nr2e1', 'Unc5a', 'Rgs16', 'Kcnh3', 'Celsr3']

    gaba_genes_2 = ['Sst', 'Chodl', 'Nos1', 'Mme', 'Tac1', 'Tacr3', 'Calb2',
                    'Nr2f2', 'Myh8', 'Tac2', 'Hpse', 'Crchr2', 'Crh', 'Esm1',
                    'Rxfp1', 'Nts', 'Pvalb', 'Gabrg1', 'Th', 'Calb1',
                    'Akr1c18', 'Sea3e', 'Gpr149', 'Reln', 'Tpbg', 'Cpne5',
                    'Vipr2', 'Nkx2-1', 'Lrrtm3', 'Clstn3', 'Nlgn2', 'Cbln3',
                    'Lrrtm2', 'Nxph1', 'Nxph2', 'Nxph4', 'Syt2', 'Hapln4',
                    'St6galnac5', 'Etv6', 'Iqgap2', 'Rasgef1b', 'Oxtr', 'Lama4',
                    'Lipa', 'Sirt4']

    for g in glutam_genes:
        glutam_ind = np.append(glutam_ind, np.array([i for i, item in enumerate(gene_id) if g == item]))

    glutam_gene_ind = list(map(int, glutam_ind))

    for g in gaba_genes_1:
        gaba_ind_1 = np.append(gaba_ind_1, np.array([i for i, item in enumerate(gene_id) if g == item]))
    gaba_gene_ind_1 = list(map(int, gaba_ind_1))

    for g in gaba_genes_2:
        gaba_ind_2 = np.append(gaba_ind_2, np.array([i for i, item in enumerate(gene_id) if g == item]))
    gaba_gene_ind_2 = list(map(int, gaba_ind_2))

    gene_indx = np.concatenate((glutam_gene_ind, gaba_gene_ind_1, gaba_gene_ind_2))
    if n_genes > 0:
        gene_index = np.unique(np.concatenate((np.array(range(n_genes)), gene_indx)))
    else:
        gene_index = np.unique(np.concatenate((np.array(range(len(gene_id))), gene_indx)))

    return gene_index



def get_data(data, batch_size, training=True, n_feature=0, gene_id=[], ref_genes=False, eps=1e-1, tau=0.11, min_num=10):
    """
    Load data from file
    input args
        data (dict): data dictionary
        batch_size (int): batch size
        training (bool): training or testing
        n_feature (int, optional): number of features
        gene_id (list, optional): list of gene IDs
        ref_genes (bool, optional): reference genes
        eps (float, optional): epsilon value
        tau (float, optional): temperature value
        min_num (int, optional): minimum number of cells per cell type

    return
        dataloader (DataLoader): data loader

    """

    print(data['log1p'].shape, len(data['cluster_label']), len(data['gene_id']))

    data_bin = np.where(data['log1p'] > eps, 1, 0)
    data_troch = torch.FloatTensor(data['log1p'])
    data_bin_troch = torch.FloatTensor(data_bin)
    tensor_data = TensorDataset(data_troch, data_bin_troch)
    print('... Done!')

    # Create dataloader.
    if training:
        dataloader = DataLoader(tensor_data, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        dataloader = DataLoader(tensor_data, batch_size=batch_size, shuffle=False, drop_last=True)

    return dataloader

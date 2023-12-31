import torch
import pickle
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F
from sklearn.metrics.cluster import adjusted_rand_score
import torch.nn.utils.prune as prune
import time, glob
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, TensorDataset
from scipy.optimize import linear_sum_assignment
from operator import itemgetter
import matplotlib.pyplot as plt
from matplotlib import gridspec, cm
from sklearn.model_selection import train_test_split
from utils.augmentation.udagan import *
from utils.nn_model import cpl_mixVAE


class train_cplmixVAE:

    def __init__(self, saving_folder='', aug_file='', n_feature=0, device=None, eps=1e-8, save_flag=True):

        self.eps = eps
        self.save = save_flag
        self.folder = saving_folder
        self.aug_file = aug_file
        self.device = device
        self.n_feature = n_feature

        if device is None:
            self.gpu = False
            print('using CPU ...')
        else:
            self.gpu = True
            torch.cuda.set_device(device)
            gpu_device = torch.device('cuda:' + str(device))
            print('using GPU ' + torch.cuda.get_device_name(torch.cuda.current_device()))

        if self.aug_file:
            self.aug_model = torch.load(self.aug_file)
            self.aug_param = self.aug_model['parameters']
            self.netA = Augmenter(noise_dim=self.aug_param['num_n'],
                                latent_dim=self.aug_param['num_z'],
                                n_zim=self.aug_param['n_zim'],
                                input_dim=self.aug_param['n_features'])
            # Load the trained augmenter weights
            self.netA.load_state_dict(self.aug_model['netA'])

            if self.gpu:
                self.netA = self.netA.cuda(self.device)

    def data_gen(self, dataset, train_size):

        test_size = dataset.shape[0] - train_size
        train_cpm, test_cpm, train_ind, test_ind = train_test_split(dataset[:, self.index], np.arange(dataset.shape[0]),
                                                                    train_size=train_size, test_size=test_size,
                                                                    random_state=0)

        return train_cpm, test_cpm, train_ind, test_ind

    def getdata(self, dataset, label=[], index=[], batch_size=128, train_size=0.9):

        self.batch_size = batch_size
        kwargs = {'pin_memory': True, 'shuffle': True, 'batch_size': batch_size}  # 'num_workers': 1,

        if len(index) > 0:
            self.index = index
        else:
            self.index = np.arange(0, dataset.shape[1])

        if len(label) > 0:
            train_ind, val_ind, test_ind = [], [], []
            for ll in np.unique(label):
                indx = np.where(label == ll)[0]
                tt_size = int(train_size * sum(label == ll))
                _, _, train_subind, test_subind = self.data_gen(dataset[indx, :], tt_size)
                # train_len = len(train_subind) // 10
                # test_len = len(test_subind) // 10
                train_ind.append(indx[train_subind])
                test_ind.append(indx[test_subind])

            train_ind = np.concatenate(train_ind)
            test_ind = np.concatenate(test_ind)
            train_set = dataset[train_ind, :]
            test_set = dataset[test_ind, :]
            self.n_class = len(np.unique(label))
        else:
            tt_size = int(train_size * dataset.shape[0])
            train_set, test_set, train_ind, test_ind = self.data_gen(dataset, tt_size)

        print(train_set.shape, test_set.shape)

        train_set_torch = torch.FloatTensor(train_set)
        train_ind_torch = torch.FloatTensor(train_ind)
        train_data = TensorDataset(train_set_torch, train_ind_torch)
        train_loader = DataLoader(train_data, drop_last=True, **kwargs)

        test_set_torch = torch.FloatTensor(test_set)
        test_ind_torch = torch.FloatTensor(test_ind)
        test_data = TensorDataset(test_set_torch, test_ind_torch)
        test_loader = DataLoader(test_data, drop_last=False, **kwargs)

        data_set_troch = torch.FloatTensor(dataset)  # torch.FloatTensor(dataset[:, self.index])
        all_ind_torch = torch.FloatTensor(range(dataset.shape[0]))
        all_data = TensorDataset(data_set_troch, all_ind_torch)
        alldata_loader = DataLoader(all_data, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)

        return train_loader, test_loader, alldata_loader, train_ind, test_ind



    def init_model(self, n_categories, state_dim, input_dim, fc_dim=100, lowD_dim=5, x_drop=0.2, s_drop=0.2, lr=.001,
                   lam=1, lam_pc=1, n_arm=2, temp=1., tau=0.01, beta=1., hard=False, variational=True, ref_prior=False,
                   trained_model='', n_pr=0, momentum=.01, n_zim=1):
        """
        Initialized the deep mixture model and its optimizer.

        input args:
            fc_dim: dimension of the hidden layer.
            lowD_dim: dimension of the latent representation.
            x_drop: dropout probability at the first (input) layer.
            s_drop: dropout probability of the state variable.
            lr: the learning rate of the optimizer, here Adam.
            n_arm: int value that indicates number of arms.
            lam: coupling factor in the cpl-mixVAE model.
            tau: temperature of the softmax layers, usually equals to 1/n_categories (0 < tau <= 1).
            beta: regularizer for the KL divergence term.
            hard: a boolean variable, True uses one-hot method that is used in Gumbel-softmax, and False uses the Gumbel-softmax function.
            state_det: a boolean variable, False uses sampling.
            trained_model: the path of a pre-trained model, in case you wish to initialized the network with a pre-trained network.
            momentum: a hyperparameter for batch normalization that updates its running statistics.
        """
        self.lowD_dim = lowD_dim
        self.n_categories = n_categories
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.temp = temp
        self.n_arm = n_arm
        self.fc_dim = fc_dim
        self.ref_prior = ref_prior
        self.n_zim = n_zim
        self.model = cpl_mixVAE(input_dim=self.input_dim, fc_dim=fc_dim, n_categories=self.n_categories, state_dim=self.state_dim,
                                lowD_dim=lowD_dim, x_drop=x_drop, s_drop=s_drop, n_arm=self.n_arm, lam=lam, lam_pc=lam_pc,
                                tau=tau, beta=beta, hard=hard, variational=variational, device=self.device, eps=self.eps,
                                ref_prior=ref_prior, momentum=momentum, n_zim=self.n_zim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        if self.gpu:
            self.model = self.model.cuda(self.device)

        if len(trained_model) > 0:
            print('Load the pre-trained model')
            # if you wish to load another model for evaluation
            loaded_file = torch.load(trained_model, map_location='cpu')
            self.model.load_state_dict(loaded_file['model_state_dict'])
            self.optimizer.load_state_dict(loaded_file['optimizer_state_dict'])
            self.init = False
            self.n_pr = n_pr
        else:
            self.init = True
            self.n_pr = 0


    def load_model(self, trained_model):
        loaded_file = torch.load(trained_model, map_location='cpu')
        self.model.load_state_dict(loaded_file['model_state_dict'])

        self.current_time = time.strftime('%Y-%m-%d-%H-%M-%S')


    def run(self, train_loader, test_loader, alldata_loader, n_epoch, n_epoch_p, c_p=0, min_con=.5, max_pron_it=0, mode='MSE'):
        """
        run the training of the cpl-mixVAE with the pre-defined parameters/settings
        pcikle used for saving the file

        input args
            data_df: a data frame including 'cluster_id', 'cluster', and 'class_label'
            train_loader: train dataloader
            test_loader: test dataloader
            validation_set:
            n_epoch: number of training epoch, without pruning
            n_epoch: number of training epoch, with pruning
            min_con: minimum value of consensus among pair of arms
            temp: temperature of sampling

        return
            data_file_id: the path of the output dictionary.
        """
        # define current_time
        self.current_time = time.strftime('%Y-%m-%d-%H-%M-%S')

        # initialized saving arrays
        train_loss = np.zeros(n_epoch)
        validation_loss = np.zeros(n_epoch)
        train_loss_joint = np.zeros(n_epoch)
        train_entropy = np.zeros(n_epoch)
        train_distance = np.zeros(n_epoch)
        train_minVar = np.zeros(n_epoch)
        train_log_distance = np.zeros(n_epoch)
        train_recon = np.zeros((self.n_arm, n_epoch))
        train_loss_KL = np.zeros((self.n_arm, self.n_categories, n_epoch))
        z_train_prob = np.zeros((self.n_arm, len(train_loader.dataset), self.n_categories))
        train_class_label = np.zeros(len(train_loader.dataset))
        bias_mask = torch.ones(self.n_categories)
        weight_mask = torch.ones((self.n_categories, self.lowD_dim))
        fc_mu = torch.ones((self.state_dim, self.n_categories + self.lowD_dim))
        fc_sigma = torch.ones((self.state_dim, self.n_categories + self.lowD_dim))
        f6_mask = torch.ones((self.lowD_dim, self.state_dim + self.n_categories))

        if self.gpu:
            bias_mask = bias_mask.cuda(self.device)
            weight_mask = weight_mask.cuda(self.device)
            fc_mu = fc_mu.cuda(self.device)
            fc_sigma = fc_sigma.cuda(self.device)
            f6_mask = f6_mask.cuda(self.device)

        if self.init:
            print("Start training...")
            for epoch in range(n_epoch):
                train_loss_val = 0
                train_jointloss_val = 0
                train_dqc = 0
                log_dqc = 0
                entr = 0
                var_min = 0
                t0 = time.time()
                train_loss_rec = np.zeros(self.n_arm)
                train_KLD_cont = np.zeros((self.n_arm, self.n_categories))
                self.model.train()

                for batch_indx, (data, d_idx), in enumerate(train_loader):
                    data = Variable(data)
                    d_idx = d_idx.to(int)
                    if self.gpu:
                        data = data.cuda(self.device)
                    trans_data = []
                    tt = time.time()
                    for arm in range(self.n_arm):
                        if self.aug_file:
                            noise = torch.randn(self.batch_size, self.aug_param['num_n'], device=self.device)
                            _, gen_data = self.netA(data, noise, True, self.device)
                            if self.aug_param['n_zim'] > 1:
                                data_bin = 0. * data
                                data_bin[data > self.eps] = 1.
                                fake_data = gen_data[:, :self.aug_param['n_features']] * data_bin
                                trans_data.append(fake_data)
                            else:
                                trans_data.append(gen_data)
                        else:
                            trans_data.append(data)

                    if self.ref_prior:
                        prior_c = torch.FloatTensor(c_p[d_idx, :])
                        if self.gpu:
                            prior_c = prior_c.cuda(self.device)
                    else:
                        prior_c = 0

                    self.optimizer.zero_grad()
                    recon_batch, p_x, r_x, x_low, qc, s, c, mu, log_var, log_qc = self.model(x=trans_data, temp=self.temp)
                    loss, loss_rec, loss_joint, entropy, dist_c, d_qc, KLD_cont, min_var_0, loglikelihood = \
                        self.model.loss(recon_batch, p_x, r_x, trans_data, mu, log_var, qc, c, prior_c, mode)
                    loss.backward()
                    self.optimizer.step()
                    train_loss_val += loss.data.item()
                    train_jointloss_val += loss_joint
                    train_dqc += d_qc
                    log_dqc += dist_c
                    entr += entropy
                    var_min += min_var_0.data.item()

                    for arm in range(self.n_arm):
                        train_loss_rec[arm] += loss_rec[arm].data.item()

                train_loss[epoch] = train_loss_val / (batch_indx + 1)
                train_loss_joint[epoch] = train_jointloss_val / (batch_indx + 1)
                train_distance[epoch] = train_dqc / (batch_indx + 1)
                train_entropy[epoch] = entr / (batch_indx + 1)
                train_log_distance[epoch] = log_dqc / (batch_indx + 1)
                train_minVar[epoch] = var_min / (batch_indx + 1)

                for arm in range(self.n_arm):
                    train_recon[arm, epoch] = train_loss_rec[arm] / (batch_indx + 1)
                    for cc in range(self.n_categories):
                        train_loss_KL[arm, cc, epoch] = train_KLD_cont[arm, cc] / (batch_indx + 1)

                print('====> Epoch:{}, Total Loss: {:.4f}, Loss_arm1: {'':.4f}, Joint Loss: {:.4f}, '
                      'Entropy: {:.4f}, d_logqz: {:.4f}, d_qz: {:.4f}, var_min: {:.4f}, Elapsed Time:{:.2f}'.format(
                    epoch, train_loss[epoch], train_recon[0, epoch], train_loss_joint[epoch],
                    train_entropy[epoch], train_log_distance[epoch], train_distance[epoch], train_minVar[epoch], time.time() - t0))

                # validation
                self.model.eval()
                with torch.no_grad():
                    val_loss_rec = 0.
                    val_loss = 0.
                    for batch_indx, (data_val, d_idx), in enumerate(test_loader):
                        d_idx = d_idx.to(int)
                        if self.gpu:
                            data_val = data_val.cuda(self.device)
                        trans_val_data = []
                        for arm in range(self.n_arm):
                           trans_val_data.append(data_val)

                        if self.ref_prior:
                            prior_c = torch.FloatTensor(c_p[d_idx, :])
                            if self.gpu:
                                prior_c = prior_c.cuda(self.device)
                        else:
                            prior_c = 0

                        recon_batch, p_x, r_x, x_low, qc, s, c, mu, log_var, _ = self.model(x=trans_val_data, temp=self.temp, eval=True)
                        loss, loss_rec, loss_joint, _, _, _, _, _, _ = self.model.loss(recon_batch, p_x, r_x, trans_val_data, mu, log_var, qc, c, prior_c, mode)
                        val_loss += loss.data.item()
                        for arm in range(self.n_arm):
                            val_loss_rec += loss_rec[arm].data.item()

                validation_loss[epoch] = val_loss_rec / (batch_indx + 1) / self.n_arm
                # total_val_loss[epoch] = val_loss / (batch_indx + 1)
                print('====> Validation Loss: {:.4f}'.format(validation_loss[epoch]))

            if self.save and n_epoch > 0:
                trained_model = self.folder + '/model/cpl_mixVAE_model_before_pruning_' + self.current_time + '.pth'
                torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, trained_model)
                bias = self.model.fcc[0].bias.detach().cpu().numpy()
                mask = range(len(bias))
                prune_indx = []
                # plot the learning curve of the network
                fig, ax = plt.subplots()
                ax.plot(range(n_epoch), train_loss, label='Training')
                ax.plot(range(n_epoch), validation_loss, label='Validation')
                ax.set_xlabel('# epoch', fontsize=16)
                ax.set_ylabel('loss value', fontsize=16)
                ax.set_title('Learning curve of the cpl-mixVAE for K=' + str(self.n_categories) + ' and S=' + str(self.state_dim))
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.legend()
                ax.figure.savefig(self.folder + '/model/learning_curve_before_pruning_K_' + str(self.n_categories) + '_' + self.current_time + '.png')
                plt.close("all")


        ind = []
        if n_epoch_p > 0:
            stop_prune = False
            if self.n_pr > 0:
                # initialized pruning parameters of the layer of the discrete variable
                bias = self.model.fcc[0].bias.detach().cpu().numpy()
                pruning_mask = np.where(bias != 0.)[0]
                prune_indx = np.where(bias == 0.)[0]
                ind = np.where(bias == 0.)[0]

        else:
            stop_prune = True

        pr = self.n_pr

        while not stop_prune:
            predicted_label = np.zeros((self.n_arm, len(train_loader.dataset)))

            # Assessment over all dataset
            self.model.eval()
            with torch.no_grad():
                for i, (data, d_idx) in enumerate(train_loader):
                    data = Variable(data)
                    d_idx = d_idx.to(int)
                    if self.gpu:
                        data = data.cuda(self.device)

                    trans_data = []
                    for arm in range(self.n_arm):
                        trans_data.append(data)

                    if self.ref_prior:
                        prior_c = torch.FloatTensor(c_p[d_idx, :])
                        if self.gpu:
                            prior_c = prior_c.cuda(self.device)
                    else:
                        prior_c = 0

                    recon, p_x, r_x, x_low, z_category, state, z_smp, mu, log_sigma, _ = self.model(trans_data, self.temp, mask=pruning_mask, eval=True)

                    for arm in range(self.n_arm):
                        z_encoder = z_category[arm].cpu().data.view(z_category[arm].size()[0], self.n_categories).detach().numpy()
                        predicted_label[arm, i * self.batch_size:min((i + 1) * self.batch_size, len(alldata_loader.dataset))] = np.argmax(z_encoder, axis=1)

            c_agreement = []
            for arm_a in range(self.n_arm):
                pred_a = predicted_label[arm_a, :]
                for arm_b in range(arm_a + 1, self.n_arm):
                    pred_b = predicted_label[arm_b, :]
                    armA_vs_armB = np.zeros((self.n_categories, self.n_categories))

                    for samp in range(pred_a.shape[0]):
                        armA_vs_armB[np.int(pred_a[samp]), np.int(pred_b[samp])] += 1

                    num_samp_arm = []
                    for ij in range(self.n_categories):
                        sum_row = armA_vs_armB[ij, :].sum()
                        sum_column = armA_vs_armB[:, ij].sum()
                        num_samp_arm.append(max(sum_row, sum_column))

                    armA_vs_armB = np.divide(armA_vs_armB, np.array(num_samp_arm), out=np.zeros_like(armA_vs_armB),
                                             where=np.array(num_samp_arm) != 0)
                    c_agreement.append(np.diag(armA_vs_armB))
                    ind_sort = np.argsort(c_agreement[-1])
                    plt.figure()
                    plt.imshow(armA_vs_armB[:, ind_sort[::-1]][ind_sort[::-1]], cmap='binary')
                    plt.colorbar()
                    plt.xlabel('arm_' + str(arm_a), fontsize=20)
                    plt.xticks(range(self.n_categories), range(self.n_categories))
                    plt.yticks(range(self.n_categories), range(self.n_categories))
                    plt.ylabel('arm_' + str(arm_b), fontsize=20)
                    plt.xticks([])
                    plt.yticks([])
                    plt.title('|c|=' + str(self.n_categories), fontsize=20)
                    plt.savefig(self.folder + '/consensus_' + str(pr) + '_arm_' + str(arm_a) + '_arm_' + str(arm_b) + '.png', dpi=600)

            c_agreement = np.mean(c_agreement, axis=0)
            agreement = c_agreement[pruning_mask]
            if (np.min(agreement) <= min_con) and pr < max_pron_it:
                if pr > 0:
                    ind_min = pruning_mask[np.argmin(agreement)]
                    ind_min = np.array([ind_min])
                    ind = np.concatenate((ind, ind_min))
                else:
                    ind_min = pruning_mask[np.argmin(agreement)]
                    if len(prune_indx) > 0:
                        ind_min = np.array([ind_min])
                        ind = np.concatenate((prune_indx, ind_min))
                    else:
                        ind.append(ind_min)
                    ind = np.array(ind)

                ind = ind.astype(int)
                print(ind)
                bias_mask[ind] = 0.
                weight_mask[ind, :] = 0.
                fc_mu[:, self.lowD_dim + ind] = 0.
                fc_sigma[:, self.lowD_dim + ind] = 0.
                f6_mask[:, ind] = 0.
                stop_prune = False
            else:
                print('No more pruning!')
                stop_prune = True

            if not stop_prune:
                print("Training with pruning...")
                bias = bias_mask.detach().cpu().numpy()
                pruning_mask = np.where(bias != 0.)[0]
                train_loss = np.zeros(n_epoch_p)
                validation_loss = np.zeros(n_epoch_p)
                total_val_loss = np.zeros(n_epoch_p)
                train_loss_joint = np.zeros(n_epoch_p)
                train_entropy = np.zeros(n_epoch_p)
                train_distance = np.zeros(n_epoch_p)
                train_minVar = np.zeros(n_epoch_p)
                train_log_distance = np.zeros(n_epoch_p)
                train_recon = np.zeros((self.n_arm, n_epoch_p))
                train_loss_KL = np.zeros((self.n_arm, self.n_categories, n_epoch_p))

                for arm in range(self.n_arm):
                    prune.custom_from_mask(self.model.fcc[arm], 'weight', mask=weight_mask)
                    prune.custom_from_mask(self.model.fcc[arm], 'bias', mask=bias_mask)
                    prune.custom_from_mask(self.model.fc_mu[arm], 'weight', mask=fc_mu)
                    prune.custom_from_mask(self.model.fc_sigma[arm], 'weight', mask=fc_sigma)
                    prune.custom_from_mask(self.model.fc6[arm], 'weight', mask=f6_mask)

                for epoch in range(n_epoch_p):
                    # training
                    train_loss_val = 0
                    train_jointloss_val = 0
                    train_dqz = 0
                    log_dqz = 0
                    entr = 0
                    var_min = 0
                    t0 = time.time()
                    train_loss_rec = np.zeros(self.n_arm)
                    train_KLD_cont = np.zeros((self.n_arm, self.n_categories))
                    ti = np.zeros(len(train_loader))
                    self.model.train()
                    # training
                    for batch_indx, (data, d_idx), in enumerate(train_loader):
                        # for data in train_loader:
                        data = Variable(data)
                        d_idx = d_idx.to(int)
                        if self.gpu:
                            data = data.cuda(self.device)

                        data_bin = 0. * data
                        data_bin[data > 0.] = 1.
                        trans_data = []
                        origin_data = []
                        trans_data.append(data)
                        tt = time.time()
                        w_param, bias_param, activ_param = 0, 0, 0
                        for arm in range(self.n_arm-1):
                            if self.aug_file:
                                noise = torch.randn(self.batch_size, self.aug_param['num_n'], device=self.device)
                                _, gen_data = self.netA(data, noise, True, self.device)
                                if self.aug_param['n_zim'] > 1:
                                    data_bin = 0. * data
                                    data_bin[data > self.eps] = 1.
                                    fake_data = gen_data[:, :self.aug_param['n_features']] * data_bin
                                    trans_data.append(fake_data)
                                else:
                                    trans_data.append(gen_data)
                            else:
                                trans_data.append(data)

                        if self.ref_prior:
                            prior_c = torch.FloatTensor(c_p[d_idx, :])
                            if self.gpu:
                                prior_c = prior_c.cuda(self.device)
                        else:
                            prior_c = 0

                        self.optimizer.zero_grad()
                        recon_batch, p_x, r_x, x_low, qz, s, z, mu, log_var, log_qz = self.model(trans_data, self.temp, mask=pruning_mask)
                        loss, loss_rec, loss_joint, entropy, dist_z, d_qz, KLD_cont, min_var_0, _ = self.model.loss(recon_batch, p_x, r_x,
                                                                                        trans_data, mu, log_var, qz, z, prior_c, mode)

                        loss.backward()
                        self.optimizer.step()
                        ti[batch_indx] = time.time() - tt
                        train_loss_val += loss.data.item()
                        train_jointloss_val += loss_joint
                        train_dqz += d_qz
                        log_dqz += dist_z
                        entr += entropy
                        var_min += min_var_0.data.item()

                        for arm in range(self.n_arm):
                            train_loss_rec[arm] += loss_rec[arm].data.item()

                    train_loss[epoch] = train_loss_val / (batch_indx + 1)
                    train_loss_joint[epoch] = train_jointloss_val / (batch_indx + 1)
                    train_distance[epoch] = train_dqz / (batch_indx + 1)
                    train_entropy[epoch] = entr / (batch_indx + 1)
                    train_log_distance[epoch] = log_dqz / (batch_indx + 1)
                    train_minVar[epoch] = var_min / (batch_indx + 1)

                    for arm in range(self.n_arm):
                        train_recon[arm, epoch] = train_loss_rec[arm] / (batch_indx + 1)
                        for c in range(self.n_categories):
                            train_loss_KL[arm, c, epoch] = train_KLD_cont[arm, c] / (batch_indx + 1)

                    print('====> Epoch:{}, Total Loss: {:.4f}, Loss_1: {'
                          ':.4f}, Loss_2: {:.4f}, Joint Loss: {:.4f}, '
                          'Entropy: {:.4f}, d_logqz: {:.4f}, '
                          'd_qz: {:.4f}, var_min: {:.4f}, Elapsed Time:{:.2f}'.format(
                        epoch, train_loss[epoch], train_recon[0, epoch],
                        train_recon[1, epoch], train_loss_joint[epoch],
                        train_entropy[epoch], train_log_distance[epoch],
                        train_distance[epoch], train_minVar[epoch],
                        time.time() - t0))

                    # validation
                    self.model.eval()
                    with torch.no_grad():
                        val_loss_rec = 0.
                        val_loss = 0.
                        for batch_indx, (data_val, d_idx), in enumerate(test_loader):
                            d_idx = d_idx.to(int)
                            if self.gpu:
                                data_val = data_val.cuda(self.device)
                            trans_val_data = []
                            for arm in range(self.n_arm):
                                trans_val_data.append(data_val)

                            if self.ref_prior:
                                prior_c = torch.FloatTensor(c_p[d_idx, :])
                                if self.gpu:
                                    prior_c = prior_c.cuda(self.device)
                            else:
                                prior_c = 0

                            recon_batch, p_x, r_x, x_low, qc, s, c, mu, log_var, _ = self.model(x=trans_val_data, temp=self.temp,
                                                                                      eval=True, mask=pruning_mask)
                            loss, loss_rec, loss_joint, _, _, _, _, _, _ = self.model.loss(recon_batch, p_x, r_x, trans_val_data,
                                                                                           mu, log_var, qc, c, prior_c, mode)
                            val_loss += loss.data.item()
                            for arm in range(self.n_arm):
                                val_loss_rec += loss_rec[arm].data.item()

                    validation_loss[epoch] = val_loss_rec / (batch_indx + 1) / self.n_arm
                    total_val_loss[epoch] = val_loss / (batch_indx + 1)
                    print('====> Validation Loss: {:.4f}'.format(validation_loss[epoch]))

                for arm in range(self.n_arm):
                    prune.remove(self.model.fcc[arm], 'weight')
                    prune.remove(self.model.fcc[arm], 'bias')
                    prune.remove(self.model.fc_mu[arm], 'weight')
                    prune.remove(self.model.fc_sigma[arm], 'weight')
                    prune.remove(self.model.fc6[arm], 'weight')

                trained_model = self.folder + '/model/cpl_mixVAE_model_after_pruning_' + str(pr+1) + '_' + self.current_time + '.pth'
                torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, trained_model)
                # plot the learning curve of the network
                fig, ax = plt.subplots()
                ax.plot(range(n_epoch_p), train_loss, label='Training')
                ax.plot(range(n_epoch_p), total_val_loss, label='Validation')
                ax.set_xlabel('# epoch', fontsize=16)
                ax.set_ylabel('loss value', fontsize=16)
                ax.set_title('Learning curve of the cpl-mixVAE for K=' + str(self.n_categories) + ' and S=' + str(
                    self.state_dim))
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.legend()
                ax.figure.savefig(self.folder + '/model/learning_curve_after_pruning_' + str(pr+1) + '_K_' + str(
                    self.n_categories) + '_' + self.current_time + '.png')
                plt.close("all")
                pr += 1

        # Evaluate the trained model
        bias = self.model.fcc[0].bias.detach().cpu().numpy()
        pruning_mask = np.where(bias != 0.)[0]
        prune_indx = np.where(bias == 0.)[0]
        max_len = len(alldata_loader.dataset)
        state_sample = np.zeros((self.n_arm, len(alldata_loader.dataset), self.state_dim))
        state_mu = np.zeros((self.n_arm, len(alldata_loader.dataset), self.state_dim))
        state_var = np.zeros((self.n_arm, len(alldata_loader.dataset), self.state_dim))
        z_prob = np.zeros((self.n_arm, len(alldata_loader.dataset), self.n_categories))
        z_sample = np.zeros((self.n_arm, len(alldata_loader.dataset), self.n_categories))
        state_cat = np.zeros([self.n_arm, len(alldata_loader.dataset)])
        predicted_label = np.zeros((self.n_arm, len(alldata_loader.dataset)))
        x_low_all = np.zeros((self.n_arm, len(alldata_loader.dataset), self.lowD_dim))
        total_loss_val = []
        total_dist_z = []
        total_dist_qz = []
        total_loss_rec = [[] for a in range(self.n_arm)]
        total_loglikelihood = [[] for a in range(self.n_arm)]
        self.model.eval()
        with torch.no_grad():
            for i, (data, d_idx) in enumerate(alldata_loader):
                data = Variable(data)
                d_idx = d_idx.to(int)
                if self.gpu:
                    data = data.cuda(self.device)

                if self.ref_prior:
                    prior_c = torch.FloatTensor(c_p[d_idx, :])
                    if self.gpu:
                        prior_c = prior_c.cuda(self.device)
                else:
                    prior_c = 0

                trans_data = []
                for arm in range(self.n_arm):
                    trans_data.append(data)

                recon, p_x, r_x, x_low, z_category, state, z_smp, mu, log_sigma, _ = self.model(trans_data, self.temp, eval=True, mask=pruning_mask)
                loss, loss_arms, loss_joint, _, dist_z, d_qz, _, _, loglikelihood = self.model.loss(recon, p_x, r_x, trans_data, mu, log_sigma, z_category, z_smp, prior_c, mode)
                total_loss_val.append(loss.data.item())
                total_dist_z.append(dist_z.data.item())
                total_dist_qz.append(d_qz.data.item())

                for arm in range(self.n_arm):
                    total_loss_rec[arm].append(loss_arms[arm].data.item())
                    total_loglikelihood[arm].append(loglikelihood[arm].data.item())

                for arm in range(self.n_arm):
                    state_sample[arm, i * self.batch_size:min((i + 1) * self.batch_size, max_len), :] = state[arm].cpu().detach().numpy()
                    state_mu[arm, i * self.batch_size:min((i + 1) * self.batch_size, max_len), :] = mu[arm].cpu().detach().numpy()
                    state_var[arm, i * self.batch_size:min((i + 1) * self.batch_size, max_len), :] = log_sigma[arm].cpu().detach().numpy()
                    z_encoder = z_category[arm].cpu().data.view(z_category[arm].size()[0], self.n_categories).detach().numpy()
                    z_prob[arm, i * self.batch_size:min((i + 1) * self.batch_size, max_len), :] = z_encoder
                    z_samp = z_smp[arm].cpu().data.view(z_smp[arm].size()[0], self.n_categories).detach().numpy()
                    z_sample[arm, i * self.batch_size:min((i + 1) * self.batch_size, max_len), :] = z_samp
                    x_low_all[arm, i * self.batch_size:min((i + 1) * self.batch_size, max_len), :] = x_low[arm].cpu().detach().numpy()
                    l = [int(lab) for lab in d_idx.numpy()]

                    for n in range(z_encoder.shape[0]):
                        state_cat[arm, i * self.batch_size + n] = np.argmax(z_encoder[n, :]) + 1

                    label_predict = []

                    for d in range(len(l)):
                        z_cat = np.squeeze(z_encoder[d, :])
                        label_predict.append(np.argmax(z_cat) + 1)

                    predicted_label[arm, i * self.batch_size:min((i + 1) * self.batch_size, max_len)] = np.array(label_predict)

        mean_test_rec = np.zeros(self.n_arm)
        mean_total_loss_rec = np.zeros(self.n_arm)
        mean_total_loglikelihood = np.zeros(self.n_arm)

        for arm in range(self.n_arm):
            mean_total_loss_rec[arm] = np.mean(np.array(total_loss_rec[arm]))
            mean_total_loglikelihood[arm] = np.mean(np.array(total_loglikelihood[arm]))
        # save data
        data_file_id = self.folder + '/model/data_' + self.current_time

        if self.save:
            self.save_file(data_file_id,
                           state_sample=state_sample,
                           state_mu=state_mu,
                           state_var=state_var,
                           train_loss=train_loss,
                           validation_loss=validation_loss,
                           total_loss_rec=mean_total_loss_rec,
                           total_likelihood=mean_total_loglikelihood,
                           total_dist_z=np.mean(np.array(total_dist_z)),
                           total_dist_qz=np.mean(np.array(total_dist_qz)),
                           mean_test_rec=mean_test_rec,
                           predicted_label=predicted_label,
                           z_prob=z_prob,
                           z_sample=z_sample,
                           lowD_rep=x_low_all,
                           prune_indx=prune_indx)

        return data_file_id


    def eval_model(self, data_mat, c_p=[], c_onehot=[], batch_size=1000, mode='MSE'):

        data_set_troch = torch.FloatTensor(data_mat)
        indx_set_troch = torch.FloatTensor(np.arange(data_mat.shape[0]))
        all_data = TensorDataset(data_set_troch, indx_set_troch)
        self.batch_size = batch_size

        data_loader = DataLoader(all_data, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)

        self.model.eval()
        bias = self.model.fcc[0].bias.detach().cpu().numpy()
        pruning_mask = np.where(bias != 0.)[0]
        prune_indx = np.where(bias == 0.)[0]
        max_len = len(data_loader.dataset)
        recon_cell = np.zeros((self.n_arm, max_len, self.input_dim))
        p_cell = np.zeros((self.n_arm, max_len, self.input_dim))
        state_sample = np.zeros((self.n_arm, max_len, self.state_dim))
        state_mu = np.zeros((self.n_arm, max_len, self.state_dim))
        state_var = np.zeros((self.n_arm, max_len, self.state_dim))
        z_prob = np.zeros((self.n_arm, max_len, self.n_categories))
        z_sample = np.zeros((self.n_arm, max_len, self.n_categories))
        data_low = np.zeros((self.n_arm, max_len, self.lowD_dim))
        state_cat = np.zeros([self.n_arm, max_len])
        prob_cat = np.zeros([self.n_arm, max_len])
        if self.ref_prior:
            predicted_label = np.zeros((self.n_arm+1, max_len))
        else:
            predicted_label = np.zeros((self.n_arm, max_len))
        data_indx = np.zeros(max_len)
        total_loss_val = []
        total_dist_z = []
        total_dist_qz = []
        total_loss_rec = [[] for a in range(self.n_arm)]
        total_loglikelihood = [[] for a in range(self.n_arm)]

        self.model.eval()
        with torch.no_grad():
            for i, (data, data_idx) in enumerate(data_loader):
                data = Variable(data)
                data_idx = data_idx.to(int)
                if self.gpu:
                    data = data.cuda(self.device)

                if self.ref_prior:
                    prior_c = torch.FloatTensor(c_p[data_idx, :])
                    if self.gpu:
                        prior_c = prior_c.cuda(self.device)
                else:
                    prior_c = 0

                trans_data = []
                for arm in range(self.n_arm):
                    trans_data.append(data)

                recon, p_x, r_x, x_low, z_category, state, z_smp, mu, log_sigma, _ = self.model(trans_data, self.temp, eval=True, mask=pruning_mask)
                loss, loss_arms, loss_joint, _, dist_z, d_qz, _, _, loglikelihood = self.model.loss(recon, p_x, r_x, trans_data, mu, log_sigma, z_category, z_smp, prior_c, mode)
                total_loss_val.append(loss.data.item())
                total_dist_z.append(dist_z.data.item())
                total_dist_qz.append(d_qz.data.item())

                if self.ref_prior:
                    predicted_label[0, i * self.batch_size:min((i + 1) * self.batch_size, max_len)] = np.argmax(c_p[data_idx, :], axis=1) + 1

                for arm in range(self.n_arm):
                    total_loss_rec[arm].append(loss_arms[arm].data.item())
                    total_loglikelihood[arm].append(loglikelihood[arm].data.item())

                for arm in range(self.n_arm):
                    state_sample[arm, i * self.batch_size:min((i + 1) * self.batch_size, max_len), :] = state[arm].cpu().detach().numpy()
                    state_mu[arm, i * self.batch_size:min((i + 1) * self.batch_size, max_len), :] = mu[arm].cpu().detach().numpy()
                    state_var[arm, i * self.batch_size:min((i + 1) * self.batch_size, max_len), :] = log_sigma[arm].cpu().detach().numpy()
                    z_encoder = z_category[arm].cpu().data.view(z_category[arm].size()[0], self.n_categories).detach().numpy()
                    z_prob[arm, i * self.batch_size:min((i + 1) * self.batch_size, max_len), :] = z_encoder
                    z_samp = z_smp[arm].cpu().data.view(z_smp[arm].size()[0], self.n_categories).detach().numpy()
                    z_sample[arm, i * self.batch_size:min((i + 1) * self.batch_size, max_len), :] = z_samp
                    data_low[arm,  i * self.batch_size:min((i + 1) * self.batch_size, max_len), :] = x_low[arm].detach().cpu().numpy()
                    label = data_idx.numpy().astype(int)
                    data_indx[i * batch_size:min((i + 1) * batch_size, max_len)] = label
                    p_cell[arm, i * self.batch_size:min((i + 1) * self.batch_size, max_len), :] = p_x[arm].cpu().detach().numpy()
                    recon_cell[arm, i * self.batch_size:min((i + 1) * self.batch_size, max_len), :] = recon[arm].cpu().detach().numpy()

                    for n in range(z_encoder.shape[0]):
                        state_cat[arm, i * self.batch_size + n] = np.argmax(z_encoder[n, :]) + 1
                        prob_cat[arm, i * self.batch_size + n] = np.max(z_encoder[n, :])

                    if self.ref_prior:
                        predicted_label[arm+1, i * self.batch_size:min((i + 1) * self.batch_size, max_len)] = np.argmax(z_encoder, axis=1) + 1
                    else:
                        predicted_label[arm, i * self.batch_size:min((i + 1) * self.batch_size, max_len)] = np.argmax(z_encoder, axis=1) + 1

        mean_test_rec = np.zeros(self.n_arm)
        mean_total_loss_rec = np.zeros(self.n_arm)
        mean_total_loglikelihood = np.zeros(self.n_arm)

        for arm in range(self.n_arm):
            mean_total_loss_rec[arm] = np.mean(np.array(total_loss_rec[arm]))
            mean_total_loglikelihood[arm] = np.mean(np.array(total_loglikelihood[arm]))
        # save data

        data_file_id = self.folder + '/model/model_eval' #_pruning_' + str(len(prune_indx))

        d_dict = dict()
        d_dict['state_sample'] = state_sample
        d_dict['state_mu'] = state_mu
        d_dict['state_var'] = state_var
        d_dict['state_cat'] = state_cat
        d_dict['prob_cat'] = prob_cat
        d_dict['total_loss_rec'] = mean_total_loss_rec
        d_dict['total_likelihood'] = mean_total_loglikelihood
        d_dict['total_dist_z'] = np.mean(np.array(total_dist_z))
        d_dict['total_dist_qz'] = np.mean(np.array(total_dist_qz))
        d_dict['mean_test_rec'] = mean_test_rec
        d_dict['predicted_label'] = predicted_label
        d_dict['data_indx'] = data_indx
        d_dict['z_prob'] = z_prob
        d_dict['z_sample'] = z_sample
        d_dict['x_low'] = data_low
        d_dict['p_c'] = p_cell
        d_dict['recon_c'] = recon_cell
        d_dict['prune_indx'] = prune_indx

        # self.save_file(data_file_id,
        #                state_sample=state_sample,
        #                state_mu=state_mu,
        #                state_var=state_var,
        #                state_cat=state_cat,
        #                prob_cat=prob_cat,
        #                total_loss_rec=mean_total_loss_rec,
        #                total_likelihood=mean_total_loglikelihood,
        #                total_dist_z=np.mean(np.array(total_dist_z)),
        #                total_dist_qz=np.mean(np.array(total_dist_qz)),
        #                mean_test_rec=mean_test_rec,
        #                predicted_label=predicted_label,
        #                data_indx=data_indx,
        #                z_prob=z_prob,
        #                z_sample=z_sample,
        #                x_low=data_low,
        #                p_c=p_cell,
        #                recon_c=recon_cell,
        #                prune_indx=prune_indx)

        return d_dict, data_file_id


    def cluster_analysis(self, data_mat, ref_label, batch_size=1000):

        data_set_troch = torch.FloatTensor(data_mat)
        label_set_troch = torch.FloatTensor(ref_label)
        all_data = TensorDataset(data_set_troch, label_set_troch)
        n_class = len(np.unique(ref_label))

        data_loader = DataLoader(all_data, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)

        self.model.eval()
        bias = self.model.fcc[0].bias.detach().cpu().numpy()
        pruning_mask = np.where(bias != 0.)[0]
        max_len = len(data_loader.dataset)
        predicted_label = np.zeros((self.n_arm, max_len))
        class_label = np.zeros(max_len)
        category_vs_class = np.zeros((self.n_arm, n_class, self.n_categories))
        z_prob = np.zeros((self.n_arm, max_len, self.n_categories))
        z_sample = np.zeros((self.n_arm, max_len, self.n_categories))

        # Assessment over all dataset
        with torch.no_grad():
            for i, (data, labels) in enumerate(data_loader):
                data = Variable(data)
                if self.gpu:
                    data = data.cuda(self.device)

                data_bin = 0. * data
                data_bin[data > 0.] = 1.
                trans_data = []
                for arm in range(self.n_arm):
                    if self.aug_file:
                        noise = torch.randn(data.size(0), self.aug_param['num_n'])
                        if self.gpu:
                            noise = noise.cuda(self.device)
                        _, gen_data = self.netA(data, noise, True, self.device)
                        if self.aug_param['n_zim'] > 1:
                            augmented_data = gen_data[:, :self.aug_param['n_features']] * data_bin
                            trans_data.append(augmented_data)
                        else:
                            trans_data.append(gen_data)
                    else:
                        trans_data.append(data)

                recon, p_x, x_low, z_category, state, z_smp, mu, log_sigma, _ = self.model(trans_data, self.temp, eval=True, mask=pruning_mask)
                _, _, _, entropy, dist_z, d_qz, _, min_var_0, _ = self.model.loss(recon, p_x, trans_data, mu, log_sigma, z_category, z_smp, 0)
                # print(min_var_0)

                for arm in range(self.n_arm):
                    z_encoder = z_category[arm].cpu().data.view(z_category[arm].size()[0], self.n_categories).detach().numpy()
                    z_prob[arm, i * batch_size:min((i + 1) * batch_size, max_len), :] = z_encoder
                    z_samp = z_smp[arm].cpu().data.view(z_smp[arm].size()[0], self.n_categories).detach().numpy()
                    z_sample[arm, i * batch_size:min((i + 1) * batch_size, max_len), :] = z_samp
                    label = labels.numpy().astype(int)
                    class_label[i * batch_size:min((i + 1) * batch_size, max_len)] = label
                    label_predict = []
                    for d in range(len(labels)):
                        z_cat = np.squeeze(z_encoder[d, :])
                        try:
                            category_vs_class[arm, label[d] - 1, np.argmax(z_cat)] += 1
                        except:
                            stop = 1
                        label_predict.append(np.argmax(z_cat) + 1)

                    predicted_label[arm, i * batch_size:min((i + 1) * batch_size, max_len)] = np.array(label_predict)

        ari = adjusted_rand_score(class_label, predicted_label[0,:])
        z_prob_mean = np.mean(z_prob, axis=0)
        z_sample_mean = np.mean(z_sample, axis=0)
        unique_class_label = np.unique(ref_label)
        numCell_per_cluster = np.zeros((self.n_arm, category_vs_class.shape[1]))
        cluster_per_cat = np.zeros((self.n_arm, category_vs_class.shape[1], category_vs_class.shape[2]))
        conf_mat_prob = np.zeros((category_vs_class.shape[1], category_vs_class.shape[2]))
        conf_mat_smp = np.zeros((category_vs_class.shape[1], category_vs_class.shape[2]))

        for arm in range(self.n_arm):
            for c in range(n_class):
                if np.sum(category_vs_class[arm, c, :]) > 0:
                    cluster_per_cat[arm, c, :] = category_vs_class[arm, c, :] / np.sum(category_vs_class[arm, c, :])

        for i, cl in enumerate(unique_class_label):
            ind = np.where(unique_class_label == cl)[0]
            conf_mat_prob[i, :] = np.mean(z_prob_mean[ind, :], axis=0)
            conf_mat_smp[i, :] = np.mean(z_sample_mean[ind, :], axis=0)

        # save data
        data_file_id = self.folder + '/model/clustering_' + self.current_time
        self.save_file(data_file_id,
                       cluster_per_cat=cluster_per_cat,
                       conf_mat_prob=conf_mat_prob,
                       conf_mat_smp=conf_mat_smp,
                       ari=ari,
                       class_label=class_label,
                       predicted_label=predicted_label)

        return cluster_per_cat, conf_mat_prob, conf_mat_smp


    def save_file(self, fname, **kwargs):
        """
        Save data as a .p file using pickle.

        input args
            fname: the path of the pre-trained network.
            kwarg: keyword arguments for input variables e.g., x=[], y=[], etc.
        """

        f = open(fname + '.p', "wb")
        data = {}
        for k, v in kwargs.items():
            data[k] = v
        pickle.dump(data, f, protocol=4)
        f.close()

    def load_file(self, fname):
        """
        load data .p file using pickle. Make sure to use the same version of
        pcikle used for saving the file

        input args
            fname: the path of the pre-trained network.

        return
            data: a dictionary including the save dataset
        """

        data = pickle.load(open(fname + '.p', "rb"))
        return data



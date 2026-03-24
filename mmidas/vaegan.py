import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from .networks_aug import RNA_augmenter as Augmenter
from .networks_aug import RNA_discriminator as Discriminator
from .utils.augmentation import TripletLoss


class vae_gan:

    def __init__(self, saving_folder='', device=None, eps=1e-6, save_flag=True):
        """
        Initialized the cpl_mixVAE class.

        input args:
            saving_folder: a string that indicates the folder to save the model(s) and file(s).
            aug_file: a string that indicates the file of the pre-trained augmenter.
            device: computing device, either 'cpu' or 'cuda'.
            eps: a small constant value to fix computation overflow.
            save_flag: a boolean variable, if True, the model is saved.
        """

        self.eps = eps
        self.save = save_flag
        self.folder = saving_folder
        self.device = device

        if device is None:
            self.device = torch.device('cpu')
            print('---> Computional node is not assigned, using CPU!')
        else:
            try:
                torch.cuda.set_device(self.device)
                print('--->' + torch.cuda.get_device_name(torch.cuda.current_device()))
            except:
                print('---> Using CPU!')


    def init_model(self, input_dim, z_dim, noise_dim, fc_dim=100, x_drop=0.5, affine=False, momentum=0.01, trained_model=''):
        """
        Initialized the VAE-GAN model.

        input args:
            input_dim: the dimension of the input data.
            z_dim: the dimension of the latent space.
            noise_dim: the dimension of the additive noise.
            fc_dim: the dimension of the fully connected layer.
            x_drop: the dropout rate of the input layer.
            alpha: triplet loss parameter.
            lam: wieghts of the loss functions.
            initial_w: if True, the weights are initialized.
            affine: if True, the batch normalization is affine.
            n_smp: the number of augmented samples.
            momentum: the momentum of the batch normalization.
            trained_model: the pre-trained model file.
        """
        
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.noise_dim = noise_dim
        self.netA = Augmenter(
                            input_dim=input_dim, 
                            latent_dim=z_dim, 
                            noise_dim=noise_dim, 
                            fc_dim=fc_dim, 
                            p_drop=x_drop, 
                            momentum=momentum, 
                            affine=affine,
                            eps=self.eps,
                            )
        
        self.netD = Discriminator(
                                input_dim=input_dim,
                                fc_dim=fc_dim,
                                p_drop=x_drop,
                                momentum=momentum, 
                                affine=affine,
                                eps=self.eps,
                                )
        
        self.aug_param = {
                        'input_dim': input_dim, 
                        'latent_dim': z_dim, 
                        'noise_dim': noise_dim,
                        'fc_dim': fc_dim,
                        'momentum': momentum,
                        'affine': affine,
                        }

        
        self.netA = self.netA.to(self.device)
        self.netD = self.netD.to(self.device)
        
        # if len(trained_model) > 0:
        #     print('Load the pre-trained model')
        #     # if you wish to load another model for evaluation
        #     self.load_model(trained_model)
            

    def load_model(self, trained_model):
        print(f'Load the pre-trained augmenter model - {trained_model}')
        loaded_model = torch.load(trained_model, map_location='cpu', weights_only=True)
        self.aug_param = loaded_model['params']
        self.netA = Augmenter(
                            input_dim=self.aug_param['input_dim'], 
                            latent_dim=self.aug_param['latent_dim'], 
                            noise_dim=self.aug_param['noise_dim'], 
                            fc_dim=self.aug_param['fc_dim'], 
                            p_drop=0., 
                            momentum=self.aug_param['momentum'], 
                            affine=self.aug_param['affine'],
                            eps=self.eps,
                            )
        self.netA.load_state_dict(loaded_model['netA'])
        self.netA = self.netA.to(self.device)


    def train(self, dataloader, n_epoch, lr, alpha, lam, tag):
        """
        run the training of the cpl-mixVAE with the pre-defined parameters/settings
        pcikle used for saving the file

        input args
            data_loader: dataloader.
            n_epoch: number of training epoch, without pruning.
            lr: learning rate.

        return
            data_file_id: the output dictionary.
        """
        # define current_time
        self.current_time = time.strftime('%Y-%m-%d-%H-%M-%S')

        batch_size = dataloader.batch_size
        iter_num = len(dataloader)
        
        self.optimD = torch.optim.Adam([{'params': self.netD.parameters()}], lr=lr)
        self.optimA = torch.optim.Adam([{'params': self.netA.parameters()}], lr=lr)
        
        # Loss functions
        criterionD = nn.BCELoss()
        mseDist = nn.MSELoss()
        
        real_label = 1.
        fake_label = 0.
        A_losses = []
        D_losses = []

        print('-'*50)
        print('Starting training the augmenter network ...')

        for epoch in range(n_epoch):
            epoch_start_time = time.time()
            A_loss_e, D_loss_e = 0, 0
            gen_loss_e, recon_loss_e = 0, 0
            triplet_loss_e = 0
            n_adv = 0
            for _, (data, data_bin) in enumerate(dataloader, 0):
                real_data = data.to(self.device)
                real_data_bin = data_bin.to(self.device)
                # Updating the discriminator -----------------------------------
                self.optimD.zero_grad()
                # Original samples
                label = torch.full((batch_size,), real_label, device=self.device)
                _, probs_real = self.netD(real_data_bin)
                loss_real = criterionD(probs_real.view(-1), label)

                if F.relu(loss_real - np.log(2) / 2) > 0:
                    loss_real.backward()
                    optim_D = True
                else:
                    optim_D = False

                # Augmented samples
                label.fill_(fake_label)
                # noise += 0.1 * torch.sign(noise)
                _, fake_data1 = self.netA(real_data, True)
                # zeros = torch.zeros(b_size, parameters['num_n'], device=device)
                _, fake_data2 = self.netA(real_data, False)
                
                # binarizing the augmented sample
                fake_data1_bin = 0. * fake_data1
                fake_data2_bin = 0. * fake_data2
                fake_data1_bin[fake_data1 > 1e-3] = 1.
                fake_data2_bin[fake_data2 > 1e-3] = 1.
                fake_data = 1. * fake_data2

                _, probs_fake1 = self.netD(fake_data1_bin.detach())
                _, probs_fake2 = self.netD(fake_data2_bin.detach())
                loss_fake = (criterionD(probs_fake1.view(-1), label) + criterionD(probs_fake2.view(-1), label)) / 2

                if F.relu(loss_fake - np.log(2) / 2) > 0:
                    loss_fake.backward()
                    optim_D = True

                # Loss value for the discriminator
                D_loss = loss_real + loss_fake

                if optim_D:
                    self.optimD.step()
                else:
                    n_adv += 1

                # Updating the augmenter ---------------------------------------
                self.optimA.zero_grad()
                # Augmented data treated as real data
                z1, probs_fake1 = self.netD(fake_data1_bin)
                z2, probs_fake2 = self.netD(fake_data2_bin)
                # z0, _ = netD(real_data)
                label.fill_(real_label)
                gen_loss = (criterionD(probs_fake1.view(-1), label) + criterionD(probs_fake2.view(-1), label)) / 2
                triplet_loss = TripletLoss(real_data_bin.view(batch_size, -1),
                                        fake_data2_bin.view(batch_size, -1),
                                        fake_data1_bin.view(batch_size, -1),
                                        alpha, 'BCE')

                recon_loss = (F.mse_loss(fake_data, real_data, reduction='mean') + criterionD(fake_data2_bin, real_data_bin)) / 2
                
                # Loss value for the augmenter
                A_loss = lam[0] * gen_loss + lam[1] * triplet_loss + lam[2] * mseDist(z1, z2) + lam[3] * recon_loss
                A_loss.backward()
                self.optimA.step()

                A_losses.append(A_loss.data.item())
                D_losses.append(D_loss.data.item())
                A_loss_e += A_loss.data.item()
                D_loss_e += D_loss.data.item()
                gen_loss_e += gen_loss.data.item()
                recon_loss_e += recon_loss.data.item()
                triplet_loss_e += triplet_loss.data.item()

            A_loss_epoch = A_loss_e / (iter_num)
            D_loss_epoch = D_loss_e / (iter_num )
            gen_loss_epoch = gen_loss_e / (iter_num)
            recon_loss_epoch = recon_loss_e / (iter_num)
            triplet_loss_epoch = triplet_loss_e / (iter_num)

            print('=====> Epoch:{}, Generator Loss: {:.4f}, Discriminator Loss: {'
                ':.4f}, Recon Loss: {:.4f}, Trip Loss: '
                '{:.4f}, Elapsed Time:{:.2f}'.format(epoch, A_loss_epoch,
                        D_loss_epoch, recon_loss_epoch, triplet_loss_epoch,
                        time.time() - epoch_start_time))

        print("-" * 50)
        # Save trained models
        filename = f'RNA_augmenter_{tag}_{self.current_time}.pth'
        torch.save(
                    {
                    'netA': self.netA.state_dict(),
                    'netD': self.netD.state_dict(),
                    'optimD': self.optimD.state_dict(),
                    'optimA': self.optimA.state_dict(),
                    'params': self.aug_param,
                    }, 
                    self.folder / filename,
                    )

        # Plot the training losses.
        plt.figure()
        plt.title("Augmenter and Discriminator Loss Values in Training")
        plt.plot(A_losses, label="A")
        plt.plot(D_losses, label="D")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(self.folder / 'loss_curve.png')
        
        return filename
        
    
    def sample_generator(self, dataloader, noise=True, scale=1., exclude_zeros=True):
        """
        Sample from the generator network.

        input args:
            data_loader: the data loader.
            noise: if True, the noise is added to the samples.
            exclude_zeros: if True, the zeros are excluded from the augmentation process.

        return:
            augmented_samples: the generated samples.
        """
        augmented_samples = []
        
        for _, (data, data_bin) in enumerate(dataloader):
            _, samples = self.netA(data, noise, scale)
            if exclude_zeros:
                samples = samples * data_bin
            
            augmented_samples.append(samples.detach().cpu().numpy())

        return np.concatenate(augmented_samples)

            
            
            
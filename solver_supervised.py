from model import Generator
from model import Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import SimpleITK as sitk
import wandb
import math
import random
from tqdm import tqdm
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, celeba_loader, rafd_loader,ufmr_dcmr_loader,ufmr_dcmr_supervised_loader,ufmr_dcmr_supervised_valid_loader, config):
        """Initialize configurations."""
        
        #self.seed_everything(20) 
        
        self.data_parallel=config.data_parallel
        # Data loader.
        self.celeba_loader = celeba_loader
        self.rafd_loader = rafd_loader
        self.ufmr_dcmr_loader = ufmr_dcmr_loader
        self.ufmr_dcmr_supervised_loader = ufmr_dcmr_supervised_loader
        self.ufmr_dcmr_supervised_valid_loader = ufmr_dcmr_supervised_valid_loader
        
        # Model configurations.
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        self.lambda_super = config.lambda_super
        self.lambda_diffmap = config.lambda_diffmap
        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs
        self.len_attrs = len(self.selected_attrs)
        self.using_ufmr_diffmap = config.using_ufmr_diffmap
        print(self.using_ufmr_diffmap)
        
        
        self.use_diff_map = config.use_diff_map
        self.gpu_ids = config.gpu_ids
        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        if self.dataset in ['CelebA', 'RaFD', 'UFMR_DCMR_Dataset', 'UFMR_DCMR_Supervised']:
            if self.using_ufmr_diffmap == True: #concat diffmap => c_dim+1
                self.G = Generator(self.g_conv_dim, self.c_dim+1, self.g_repeat_num)
                self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num, self.using_ufmr_diffmap)  
            
            else:
                self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
                self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num, self.using_ufmr_diffmap) 
   

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
        
        if self.data_parallel == True:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.G=torch.nn.DataParallel(self.G)
            self.D=torch.nn.DataParallel(self.D)
            
        self.G.to(self.device)
        self.D.to(self.device)
        
                
        wandb.watch(self.G)
        wandb.watch(self.D)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
     

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)
    # save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
    
    def nii_image_save(self, x, save_path):
        source = sitk.GetImageFromArray(x)
    
    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=4, dataset='UFMR_DCMR_Dataset', selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        c_trg_list = []
        for i in range(c_dim):
            if dataset == 'UFMR_DCMR_Supervised':
                c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list
        

    def classification_loss(self, logit, target, dataset='CelebA'):
        """Compute binary or softmax cross entropy loss."""
        
        if dataset == 'UFMR_DCMR_Supervised':
            return F.binary_cross_entropy_with_logits(logit, target) / logit.size(0)
    def error(self, actual, predicted):
        return actual-predicted
    
    def mse(self, actual, predicted):
        mse = np.mean(np.square(self.error(actual, predicted)))
        if mse == 0:
            return 100
        return mse
    def rmse(self, actual, predicted ):
        return np.sqrt(self.mse(actual, predicted))
    
    def nrmse(self, actual, predicted):
        if actual.max() - actual.min() == 0: return 0
        return self.rmse(actual, predicted) / (actual.max() - actual.min())
    
    def psnr(self, actual, predicted ):
        mse = self.mse(actual, predicted) 
        if mse == 100 : return 100
        PIXEL_MAX = 1.0
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    
    def ssim(self, actual, predicted):
        s1 = actual.sum()
        s2 = predicted.sum()
        
        ss = (actual*actual).sum() + (predicted*predicted).sum()
        s12 = (actual*predicted).sum()
        
        vari = ss - s1*s1 - s2*s2
        covar = s12 - s1*s2
        
        ssim_c1 = .01*.01
        ssim_c2 = .03*.03
        ssim_value = (2*s1*s2 + ssim_c1) * (2*covar + ssim_c2) / ((s1*s1 + s2*s2 + ssim_c1) * (vari + ssim_c2))
        
        return ssim_value.mean()
    
    def train(self):
        """Train StarGAN within a single dataset."""
        # Set data loader.
        if self.dataset == 'UFMR_DCMR_Supervised':
            data_loader = self.ufmr_dcmr_supervised_loader
            valid_data_loader = self.ufmr_dcmr_supervised_valid_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        valid_data_iter = iter(valid_data_loader)
        #print('data_iter : ', next(data_iter))
        if self.using_ufmr_diffmap == True:
            x_fixed, x_diff_map, c_org, dcmr_image, dcmr_label  = next(valid_data_iter)
            x_fixed = torch.cat([x_fixed, x_diff_map], dim=1)
        else:
            x_fixed, c_org, dcmr_image, dcmr_label  = next(valid_data_iter)
        x_fixed= x_fixed.to(self.device)
        dcmr_fixed_image = dcmr_image
        print_dcmr = True
        
        c_fixed_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)
        print('c_fixed_list:', c_fixed_list)
        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                if self.using_ufmr_diffmap == True:
                    x_real, x_real_diff_map, label_org, dcmr_image, dcmr_label  = next(data_iter)
                    x_real = torch.cat([x_real, x_real_diff_map], dim=1)
                else:
                    x_real, label_org, dcmr_image, dcmr_label  = next(data_iter)
            except:
                data_iter = iter(data_loader)
                if self.using_ufmr_diffmap == True:
                    x_real, x_real_diff_map, label_org, dcmr_image, dcmr_label  = next(data_iter)
                    x_real = torch.cat([x_real, x_real_diff_map], dim=1)
                else:
                    x_real, label_org, dcmr_image, dcmr_label  = next(data_iter)

            cur_batch = len(x_real)
            rand_int = torch.randint(self.len_attrs, size=(cur_batch,))
            
            label_trg = torch.zeros(cur_batch,self.len_attrs)
            label_trg[range(cur_batch),rand_int]=1
            
            image_trg = []
            for time_idx, idx in enumerate(rand_int):
            #for idx, time_idx in enumerate(rand_int):
                image_trg.append(dcmr_image[idx][time_idx].tolist())
            image_trg = torch.Tensor(image_trg).to(self.device)

           
            if self.dataset == 'UFMR_DCMR_Supervised':
                c_org = label_org.clone()
                c_trg = label_trg.clone()
                

            x_real = x_real.to(self.device)           # Input images.
            c_org = c_org.to(self.device)             # Original domain labels.
            c_trg = c_trg.to(self.device)             # Target domain labels.
            label_org = label_org.to(self.device)     # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)     # Labels for computing classification loss.
            if self.using_ufmr_diffmap == True: 
                x_real_diff_map = x_real_diff_map.to(self.device)
            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src, out_cls = self.D(x_real)
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, label_org, self.dataset)

            # Compute loss with fake images.
            x_fake = self.G(x_real, c_trg)
            if self.using_ufmr_diffmap == True: 
                x_fake = torch.cat([x_fake, x_real_diff_map], dim=1)
            out_src, out_cls = self.D(x_fake.detach())
            out_cls_loss = self.classification_loss(out_cls, c_trg, self.dataset)########my problem
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            #d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * (d_loss_cls*0.3 + out_cls_loss) + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()
            wandb.log({
                    "Epoch" : i+1,
                    "D/loss_real" : d_loss_real.item(),
                    "D/loss_fake" : d_loss_fake.item(),
                    "D/loss_cls" : d_loss_cls.item(),
                    "D/loss_gp" : d_loss_gp.item(),
                })

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            
            if (i+1) % self.n_critic == 0:
                # Original-to-target domain.
                x_fake = self.G(x_real, c_trg)
                if self.using_ufmr_diffmap == True: 
                    x_fake = torch.cat([x_fake, x_real_diff_map], dim=1)
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src) #real/fake
                g_loss_cls = self.classification_loss(out_cls, label_trg, self.dataset) #domain
                print('c_trg:',c_trg)
                print('out_cls:', out_cls)
                print('label_trg:', label_trg)
                
                g_dcmr_loss = torch.mean(torch.abs(x_fake - image_trg)) # supervised loss 

                # Target-to-original domain.
                x_reconst = self.G(x_fake, c_org)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))
                
    
                # Using difference map
                g_loss_diff_map=0.0
                
                if self.use_diff_map == True:

                    diff_fake_list=torch.empty(self.len_attrs,cur_batch,1, self.image_size,self.image_size)

                    for diff_idx in range(self.len_attrs):

                        diff_label_trg = torch.zeros(cur_batch,self.len_attrs)
                        diff_label_trg[range(cur_batch),diff_idx]=1

                        x_fake_diff = self.G(x_real,diff_label_trg)

                        diff_fake_list[diff_idx] = x_fake_diff

                    diff_map = torch.empty(self.len_attrs-1, cur_batch,1, self.image_size,self.image_size)
                    real_diff_map = torch.empty(self.len_attrs-1, cur_batch,1, self.image_size,self.image_size)


                    for diff_idx in range(self.len_attrs-1):

                        diff_map[diff_idx] = diff_fake_list[diff_idx+1] - diff_fake_list[diff_idx]
                        real_diff_map[diff_idx] = dcmr_image[diff_idx+1] - dcmr_image[diff_idx]


                        for b in range(cur_batch):
                            
                            hist = torch.histc(real_diff_map[diff_idx][b], bins=21, min=-1, max=1)
                            
                            threshold_idx = torch.argmax(hist).data
                            
                            threshold = ((threshold_idx-10) * 0.1)
                                
                            zero_map = torch.zeros(self.image_size,self.image_size)
                            zero_map = torch.where(real_diff_map[diff_idx][b]>threshold*0.3, real_diff_map[diff_idx][b], zero_map)
                            real_diff_map[diff_idx][b] = torch.where(real_diff_map[diff_idx][b]<threshold, real_diff_map[diff_idx][b], zero_map)
                            
                            zero_map = torch.zeros(self.image_size,self.image_size)
                            zero_map = torch.where(diff_map[diff_idx][b]>threshold*0.3, diff_map[diff_idx][b], zero_map)
                            diff_map[diff_idx][b] = torch.where(diff_map[diff_idx][b]<threshold, diff_map[diff_idx][b], zero_map)

                        g_loss_diff_map = g_loss_diff_map + torch.mean(torch.abs(real_diff_map[diff_idx] - diff_map[diff_idx]))

                    g_loss_diff_map = g_loss_diff_map/(self.len_attrs-1)

                # Backward and optimize.
                g_loss_fake = g_loss_fake
                g_loss_rec = self.lambda_rec * g_loss_rec
                g_loss_cls = self.lambda_cls * g_loss_cls
                g_dcmr_loss = self.lambda_rec*g_dcmr_loss*self.lambda_super
                g_loss_diff_map = self.lambda_rec*g_loss_diff_map*self.lambda_diffmap
                
                g_loss = g_loss_fake + g_loss_rec + g_loss_cls +g_dcmr_loss+ g_loss_diff_map
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()
                loss['G/loss_supervised'] = g_dcmr_loss.item()
                loss['G/loss_diff_map'] = g_loss_diff_map.item()
                wandb.log({
                     "Epoch" : i+1,
                    "G/loss_fake" : g_loss_fake.item(),
                    "G/loss_rec" :g_loss_rec.item(),
                    "G/loss_cls" : g_loss_cls.item(),
                    "G/loss_supervised" : g_dcmr_loss.item(),
                    "G/loss_diff_map" :  g_loss_diff_map.item()
                })
            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    #x_fake_list = [x_fixed]
                    x_fake_list = []
                    for c_fixed in c_fixed_list:
                        x_fake_list.append(self.G(x_fixed, c_fixed))
                    
                    one_mse_dict = {}
                    one_psnr_dict = {}
                    
                    two_mse_dict = {}
                    two_psnr_dict = {}
                    
                    
                    
                    for num_sitk in range(self.len_attrs):
                        
                        dict_num = num_sitk
                        
                        if self.len_attrs <= 3:
                            dict_num = dict_num+1
                        
                            
                        
                        if print_dcmr==True:
                            
                            
                            source = sitk.GetImageFromArray(x_fixed[0].data.cpu())
                            sitk.WriteImage(source, os.path.join(self.sample_dir, 'UFMR_one_images.nii.gz'))
                            
                            source = sitk.GetImageFromArray(x_fixed[1].data.cpu())
                            sitk.WriteImage(source, os.path.join(self.sample_dir, 'UFMR_two_images.nii.gz'))
                        
                            source = sitk.GetImageFromArray(dcmr_fixed_image[num_sitk][0].data.cpu())
                            sitk.WriteImage(source, os.path.join(self.sample_dir, 'DCMR_{}-one_images.nii.gz'.format(num_sitk)))
                            
                            source = sitk.GetImageFromArray(dcmr_fixed_image[num_sitk][1].data.cpu())
                            sitk.WriteImage(source, os.path.join(self.sample_dir, 'DCMR_{}-two_images.nii.gz'.format(num_sitk)))
                            
                            
                        source = sitk.GetImageFromArray(x_fake_list[num_sitk][0].data.cpu())
                        sitk.WriteImage(source, os.path.join(self.sample_dir, str(num_sitk)+'{}-one_images.nii.gz'.format(i+1)))
                        
                        one_mse = (self.mse(np.array(dcmr_fixed_image[num_sitk][0].data.cpu()+1), np.array(x_fake_list[num_sitk][0].data.cpu()+1)))
                        one_psnr = (self.psnr(np.array(dcmr_fixed_image[num_sitk][0].data.cpu()+1), np.array(x_fake_list[num_sitk][0].data.cpu()+1)))
                        
                        new_one_mse = {"one_mse_"+str(dict_num):one_mse}
                        new_one_psnr = {"one_psnr_"+str(dict_num):one_psnr}
                        
                        source = sitk.GetImageFromArray(x_fake_list[num_sitk][1].data.cpu())
                        sitk.WriteImage(source, os.path.join(self.sample_dir, str(num_sitk)+'{}-two_images.nii.gz'.format(i+1)))
                        
                        two_mse = (self.mse(np.array(dcmr_fixed_image[num_sitk][1].data.cpu()+1), np.array(x_fake_list[num_sitk][1].data.cpu()+1)))
                        two_psnr = (self.psnr(np.array(dcmr_fixed_image[num_sitk][1].data.cpu()+1), np.array(x_fake_list[num_sitk][1].data.cpu()+1)))
                        
                        new_two_mse = {"two_mse_"+str(dict_num):two_mse}
                        new_two_psnr = {"two_psnr_"+str(dict_num):two_psnr}
                        
                        one_mse_dict.update(new_one_mse)
                        one_psnr_dict.update(new_one_psnr)
                        two_mse_dict.update(new_two_mse)
                        two_psnr_dict.update(new_two_psnr)
                    
                    print_dcmr=False
                    
#                     print('one_mse_{} : ', one_mse_list[0], one_mse_list[1], one_mse_list[2], one_mse_list[3] )
#                     print('two_mse_{} : ', two_mse_list[0], two_mse_list[1], two_mse_list[2], two_mse_list[3] )   
#                     print('one_psnr_{} : ', one_psnr_list[0], one_psnr_list[1], one_psnr_list[2], one_psnr_list[3] )
#                     print('two_psnr_{} : ', two_psnr_list[0], two_psnr_list[1], two_psnr_list[2], two_psnr_list[3] ) 
                    wandb_lod_dict={}
                    wandb_lod_dict.update(one_mse_dict)
                    wandb_lod_dict.update(one_psnr_dict)
                    wandb_lod_dict.update(two_mse_dict)
                    wandb_lod_dict.update(two_psnr_dict)
                    
                    wandb.log(wandb_lod_dict)
                    
            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))


    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        # Set data loader.
        if self.dataset == 'UFMR_DCMR_Supervised':
            data_loader = self.ufmr_dcmr_supervised_loader
        
        total_mse = np.zeros(4)
        total_psnr = np.zeros(4)
        total_nrmse = np.zeros(4)
        total_ssim = np.zeros(4)
        with torch.no_grad():
            for i, (x_real, x_diff_map, c_org, dcmr_image, dcmr_label, ufmr_filename, dcmr_filename) in tqdm(enumerate(data_loader)):

                # Prepare input images and target domain labels.
                if torch.mean(x_diff_map) != -1:
                    x_real = torch.cat([x_real, x_diff_map], dim=1)
                x_real = x_real.to(self.device)
        
                c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)
                #print(c_trg_list)
                # Translate images.
                x_fake_list = []
                for c_trg in c_trg_list:
                    x_fake_list.append(self.G(x_real, c_trg))

                # Save the translated images.
                source = sitk.GetImageFromArray(x_real[0].data.cpu())
                sitk.WriteImage(source, os.path.join(self.result_dir, '{}-'.format(i + 1) + ufmr_filename[0]))
                
                for num_sitk in range(self.len_attrs):
                    dcmr_num_filename = dcmr_filename[num_sitk]
                    dict_num = num_sitk

                    if self.len_attrs <= 3:
                        dict_num = dict_num + 1
                    
                    source = sitk.GetImageFromArray(x_fake_list[num_sitk][0].data.cpu())
                    sitk.WriteImage(source, os.path.join(self.result_dir, '{}-'.format(i + 1) + str(num_sitk) + '-fake_images.nii.gz'))
                    
                    source = sitk.GetImageFromArray(dcmr_image[num_sitk][0].data.cpu())
                    sitk.WriteImage(source, os.path.join(self.result_dir, '{}-'.format(i + 1) + str(num_sitk) + dcmr_num_filename[0]))
                    
                    
                    metric_X = np.array(dcmr_image[num_sitk][0].data.cpu())
                    metric_Y = np.array(x_fake_list[num_sitk][0].data.cpu())
                                    
                    metric_X = (metric_X+1)/2
                    metric_Y = (metric_Y+1)/2

                    mse = (self.mse(metric_X,metric_Y))
                    
                    psnr = (self.psnr(metric_X,metric_Y))
                    
                    nrmse = (self.nrmse(metric_X,metric_Y))
                    
                    ssim = (self.ssim(metric_X,metric_Y))
                    
                    if nrmse == 0:
                        print("nrmse_0 : ",dcmr_num_filename[0])
                    
                    total_mse[dict_num] = total_mse[dict_num]+mse
                    total_psnr[dict_num] = total_psnr[dict_num]+psnr
                    total_nrmse[dict_num] = total_nrmse[dict_num]+nrmse
                    total_ssim[dict_num] = total_ssim[dict_num]+ssim
                    
                
                    
            total_mse = total_mse/len(data_loader)
            total_psnr = total_psnr/len(data_loader)
            total_nrmse = total_nrmse/len(data_loader)
            total_ssim = total_ssim/len(data_loader)
                #x_concat = torch.cat(x_fake_list, dim=3)
#                 result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
#                 save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                #print('Saved real and fake images into {}...'.format(result_path))
            print('total_mse:', total_mse)
            print('total_psnr', total_psnr)
            print('total_nrmse', total_nrmse)
            print('total_ssim', total_ssim)



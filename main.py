import os
import argparse
from solver_supervised import Solver
from data_loader import get_loader, get_valid_loader
from torch.backends import cudnn
#import wandb
import random
import numpy as np
import torch 
import wandb
def str2bool(v):
    return v.lower() in ('true')

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    #print(random.random())
    if torch.cuda.is_available(): 
        print(f'seed : {seed_value}')
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed = 20
seed_everything(seed)

def main(config):
    # For fast training.
    cudnn.benchmark = True

    config.log_dir = config.log_dir+config.trial
    config.model_save_dir = config.model_save_dir+config.trial
    config.sample_dir = config.sample_dir+config.trial
    config.result_dir = config.result_dir+config.trial
    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Data loader.
    celeba_loader = None
    rafd_loader = None
    ufmr_dcmr_loader = None
    ufmr_dcmr_supervised_loader = None
    ufmr_dcmr_supervised_valid_loader = None

    if config.dataset in ['UFMR_DCMR_Supervised']:
        ufmr_dcmr_supervised_loader = get_loader(config.UFMR_DCMR_image_dir, config.attr_path, config.selected_attrs,
                                   config.celeba_crop_size, config.image_size, config.batch_size,
                                   'UFMR_DCMR_Supervised', config.mode, config.num_workers,config.using_ufmr_diffmap)
        ufmr_dcmr_supervised_valid_loader = get_valid_loader(config.UFMR_DCMR_image_dir, config.attr_path, config.selected_attrs,
                                   config.celeba_crop_size, config.image_size, 2,
                                   'UFMR_DCMR_Supervised', 'train_valid', 1,config.using_ufmr_diffmap)

    
    
    #print(ufmr_dcmr_loader)
    # Solver for training and testing StarGAN.
    solver = Solver(celeba_loader, rafd_loader, ufmr_dcmr_loader,ufmr_dcmr_supervised_loader,ufmr_dcmr_supervised_valid_loader, config)
#     wandb.watch(solver)
    
    if config.mode == 'train':
        if config.dataset in ['CelebA', 'RaFD', 'UFMR_DCMR_Dataset','UFMR_DCMR_Supervised']:
            solver.train()
        elif config.dataset in ['Both']:
            solver.train_multi()
    elif config.mode == 'test':
        if config.dataset in ['CelebA', 'RaFD', 'UFMR_DCMR_Dataset','UFMR_DCMR_Supervised']:
            solver.test()
        elif config.dataset in ['Both']:
            solver.test_multi()


if __name__ == '__main__':
    
    
    
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=4, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--c2_dim', type=int, default=8, help='dimension of domain labels (2nd dataset)')
    parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
    parser.add_argument('--rafd_crop_size', type=int, default=256, help='crop size for the RaFD dataset')
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--lambda_super', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_diffmap', type=float, default=10, help='weight for reconstruction loss')
    
    # Training configuration.
    parser.add_argument('--dataset', type=str, default='UFMR_DCMR_Supervised', choices=['CelebA', 'RaFD', 'UFMR_DCMR_Dataset','UFMR_DCMR_Supervised','Both'])
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=100000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=20000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['Pre', 'Early', '90s', 'Delay'])
    parser.add_argument('--use_diff_map', type=bool, default=False)
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3')
    
    parser.add_argument('--using_ufmr_diffmap', type=bool, default=False)


    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)

    # Directories.
    parser.add_argument('--UFMR_DCMR_image_dir', type=str, default='/data/DCMR_UFMR/Split_Data')
    parser.add_argument('--celeba_image_dir', type=str, default='data/celeba/images')
    parser.add_argument('--attr_path', type=str, default='/data/DCMR_UFMR/Split_Data')
    parser.add_argument('--rafd_image_dir', type=str, default='data/RaFD/train')
    parser.add_argument('--log_dir', type=str, default='/model/logs')
    parser.add_argument('--model_save_dir', type=str, default='/model/models')
    parser.add_argument('--sample_dir', type=str, default='/model/samples')
    parser.add_argument('--result_dir', type=str, default='/model/results')
    parser.add_argument('--trial', type=str )
    parser.add_argument('--data_parallel', type=bool )
    
    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)
    
    config = parser.parse_args()
    
    wandb.init(project='UFMR_DCMR')
    wandb.run.name = config.trial
    
    
    print(config)
    main(config)
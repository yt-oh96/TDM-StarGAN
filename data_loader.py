from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import torchio as tio
import numpy as np

import skimage.transform as skTrans

class UFMR_DCMR_Supervised(data.Dataset):
    """Dataset class for the UFMR_DCMR_Supervised dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode, image_size,using_ufmr_diffmap):
        """Initialize and preprocess the UFMR_DCMR_Supervised dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path # /data/SMC_data_210324_DCMR-UFMR/UFMR_DCMR_crop_resample_tanhNorm/Split_Data
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.train_valid_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.image_size = image_size
        
        self.len_attrs = len(self.selected_attrs)
        self.using_ufmr_diffmap = using_ufmr_diffmap
        
        
        self.preprocess()
        
        if self.mode == 'train':
            self.num_images = len(self.train_dataset)
            print('train_len:',self.num_images)
        elif self.mode =='train_valid':
            self.num_images = len(self.train_valid_dataset)
            print('train_valid_dataset:',self.num_images)
        elif self.mode == 'test':
            self.num_images = len(self.test_dataset)
            print('test_len:',self.num_images)
            
    def dataset_maker(self, version):
        
        
        attr_csv = os.path.join(self.attr_path, version+'_label.csv')#only ufmr data name
        
        lines = [line.rstrip() for line in open(attr_csv, 'r')]
        all_attr_names = lines[0].split(',')[3:]
        print(all_attr_names)
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name
        
        print( self.attr2idx)
        lines = lines[1:]
        random.shuffle(lines)
        
        for i, line in enumerate(lines):
            split = line.split(',')
            filename = split[0] # ex)UFMR_499_volume_17_slice_41.nii.gz
            
            if self.len_attrs == 4:
                #values = split[3:] # [-1 1 -1 -1]
                values = [-1,-1,-1,-1]
            elif self.len_attrs == 3:
                #values = split[4:] # [-1(x) 1 -1 -1]
                values = [-1,-1,-1]
            elif self.len_attrs == 2:
                #values = split[4:-1] # [-1(x) 1 -1 -1(x)]
                values = [-1,-1]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx-1] == '1') #[0 1 0 0]
            
            new_filename = filename.split('_') 
            new_filename[0] = 'DCMR' #DCMR, 499, volume, 17, slice, 41.nii.gz
            
            dcmr_filename_dict = {}
            for time in range(4):
                if self.len_attrs <= 3 and time==0:
                    continue
                if self.len_attrs == 2 and time==3:
                    continue
                    
                    
                new_filename[3] = str(time)
                dcmr_filename = '_'.join(new_filename) # DCMR_499_volume_0_slice_41.nii.gz ... DCMR_499_volume_3_slice_41.nii.gz
                dcmr_filename_dict['dcmr_filename'+str(time)] = dcmr_filename
                
            data_dict = {}
            
            
            data_dict['ufmr_filename'] = filename
            data_dict['ufmr_label'] = label
            
            #data_dict['dcmr0_filename'] = dcmr_filename[0]
            data_dict['dcmr1_filename'] = dcmr_filename_dict['dcmr_filename1']
            data_dict['dcmr2_filename'] = dcmr_filename_dict['dcmr_filename2']
            #data_dict['dcmr3_filename'] = dcmr_filename[3]


            #data_dict['dcmr0_label'] = [True,False,False,False]
            
            #data_dict['dcmr3_label'] = [False,False,False,True]
            
            if self.len_attrs >= 3:
                data_dict['dcmr3_filename'] = dcmr_filename_dict['dcmr_filename3']
                data_dict['dcmr1_label'] = [True,False,False]
                data_dict['dcmr2_label'] = [False,True,False]
                data_dict['dcmr3_label'] = [False,False,True]
            if self.len_attrs == 4:
                data_dict['dcmr0_filename'] = dcmr_filename_dict['dcmr_filename0']
                data_dict['dcmr0_label'] = [True,False,False,False]
                data_dict['dcmr1_label'] = [False,True,False,False]
                data_dict['dcmr2_label'] = [False,False,True,False]
                data_dict['dcmr3_label'] = [False,False,False,True]
                
            
            if version == 'train':   
                self.train_dataset.append(data_dict)
            elif version == 'train_valid':
                self.train_valid_dataset.append(data_dict)
            elif version =='test' :
                self.test_dataset.append(data_dict)
                
                
          

    def preprocess(self):
        """Preprocess the UFMR_DCMR attribute file."""
        
        if self.mode == 'train':
            self.dataset_maker(version='train')
            print('train:',len(self.train_dataset))
        elif self.mode == 'train_valid':
            self.dataset_maker(version='train_valid')
            print('train_valid:',len(self.train_valid_dataset))
        elif self.mode == 'test':
            self.dataset_maker(version='test')
            print('test:',len(self.test_dataset))

        print('Finished preprocessing the UFMR_DCMR dataset...')
        
  

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        #dcmr0_filename
        #dcmr0_label
        if self.mode == 'train':
            dataset = self.train_dataset
            mode_dir_name = '/Train'
        elif self.mode == 'train_valid':
            dataset = self.train_valid_dataset
            mode_dir_name = '/Validation'
        elif self.mode == 'test':
            dataset = self.test_dataset
            mode_dir_name ='/Test'
        
        dcmr_image_list=[]
        dcmr_label_list=[]
        dcmr_filename_list = []
        random_h = random.randrange(0,2)
        random_h = 1
        if self.len_attrs == 4:
            dcmr0_filename, dcmr0_label = dataset[index]['dcmr0_filename'], dataset[index]['dcmr0_label']
            
            dcmr0_image = tio.ScalarImage(os.path.join(self.image_dir+mode_dir_name, dcmr0_filename)).data
            dcmr0_image = self.data_setting(dcmr0_image, random_h)
            
            
            dcmr_filename_list.append(dcmr0_filename)
            dcmr_image_list.append(dcmr0_image)
            dcmr_label_list.append(torch.FloatTensor(dcmr0_label))
        
        ufmr_filename, ufmr_label   = dataset[index]['ufmr_filename'], dataset[index]['ufmr_label']
        dcmr1_filename, dcmr1_label = dataset[index]['dcmr1_filename'], dataset[index]['dcmr1_label']
        dcmr2_filename, dcmr2_label = dataset[index]['dcmr2_filename'], dataset[index]['dcmr2_label']
        
        ufmr_image = tio.ScalarImage(os.path.join(self.image_dir+mode_dir_name, ufmr_filename)).data
        dcmr1_image = tio.ScalarImage(os.path.join(self.image_dir+mode_dir_name, dcmr1_filename)).data
        dcmr2_image = tio.ScalarImage(os.path.join(self.image_dir+mode_dir_name, dcmr2_filename)).data
         
        
        ufmr_image = self.data_setting(ufmr_image, random_h)
        dcmr1_image = self.data_setting(dcmr1_image, random_h)
        dcmr2_image = self.data_setting(dcmr2_image, random_h)
        
        dcmr_filename_list.append(dcmr1_filename)
        dcmr_image_list.append(dcmr1_image)
        dcmr_label_list.append(torch.FloatTensor(dcmr1_label))
        
        dcmr_filename_list.append(dcmr2_filename)
        dcmr_image_list.append(dcmr2_image)
        dcmr_label_list.append(torch.FloatTensor(dcmr2_label))
        
        if self.len_attrs >= 3:
            dcmr3_filename, dcmr3_label = dataset[index]['dcmr3_filename'], dataset[index]['dcmr3_label']
            dcmr3_image = tio.ScalarImage(os.path.join(self.image_dir+mode_dir_name, dcmr3_filename)).data
            dcmr3_image = self.data_setting(dcmr3_image, random_h)
            
            dcmr_filename_list.append(dcmr3_filename)
            dcmr_image_list.append(dcmr3_image)
            dcmr_label_list.append(torch.FloatTensor(dcmr3_label))
        
        if self.using_ufmr_diffmap==True:
            ufmr_diff_filename = 'diffmap_'+ufmr_filename
            ufmr_diff_image = tio.ScalarImage(os.path.join(self.image_dir+'/Diff_map', ufmr_diff_filename)).data
            ufmr_diff_image = self.data_setting(ufmr_diff_image, random_h)
        
        
        if self.mode == 'test':
            if self.using_ufmr_diffmap==True:
                return ufmr_image, ufmr_diff_image, torch.FloatTensor(ufmr_label), dcmr_image_list, dcmr_label_list, ufmr_filename, dcmr_filename_list
            ufmr_diff_image = torch.zeros(ufmr_image.shape) - torch.ones(ufmr_image.shape)
            return ufmr_image, ufmr_diff_image, torch.FloatTensor(ufmr_label), dcmr_image_list, dcmr_label_list, ufmr_filename, dcmr_filename_list
        
        
        if self.using_ufmr_diffmap==True:
            return ufmr_image, ufmr_diff_image, torch.FloatTensor(ufmr_label), dcmr_image_list, dcmr_label_list
        
        return ufmr_image, torch.FloatTensor(ufmr_label), dcmr_image_list, dcmr_label_list
                                
        #return ufmr_image, torch.FloatTensor(label), [dcmr0_image, dcmr1_image, dcmr2_image, dcmr3_image], [torch.FloatTensor(dcmr0_label), torch.FloatTensor(dcmr1_label), torch.FloatTensor(dcmr2_label), torch.FloatTensor(dcmr3_label)]
    
    def data_setting(self, image, random_h):
        if random_h==0:
            image = image[:,:int(image.shape[1]/2) , :, :] #C,W,H,D
        else:
            image = image[:,int(image.shape[1]/2): , :, :]
            
        #image = np.expand_dims(image, axis=3) #C,W,H,D
        #image = self.transform(image)# tio transform
        image = np.squeeze(image, axis=0)
        #image = np.transpose(image, (2, 1, 0))   #D,H,W  == C,H,W   
        
        image = skTrans.resize(image, (self.image_size, self.image_size,1), order=1, preserve_range=True)
       
        image = self.transform(image)
        #print(image.shape)
        return image.float()

    def __len__(self):
        """Return the number of images."""
        return self.num_images
    
def get_tio_transform(opt, params=None, convert=True):
    transform_list = []
    
    if 'tio_CropOrPad' in opt.preprocess:
        transform_list.append(tio.CropOrPad(
            (opt.crop_size, opt.crop_size,1))
                             )
        
        #transform_list.append(tio.)
    if convert:
        transform_list.append(tio.RescaleIntensity(out_min_max=(-1,1)))
        
    
    return tio.Compose(transform_list)

def get_loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128, 
               batch_size=16, dataset='UFMR_DCMR_Dataset', mode='train', num_workers=1,using_ufmr_diffmap=False):
    """Build and return a data loader."""
    transform = []
    transform.append(T.ToTensor())
    transform = T.Compose(transform)
    
    if dataset == 'UFMR_DCMR_Supervised':
        dataset = UFMR_DCMR_Supervised(image_dir, attr_path, selected_attrs, transform, mode,image_size,using_ufmr_diffmap)
    
    print(len(dataset))
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader

def get_valid_loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128, 
               batch_size=16, dataset='UFMR_DCMR_Dataset', mode='train_valid', num_workers=1,using_ufmr_diffmap=False):
    """Build and return a data loader."""
    transform = []
    transform.append(T.ToTensor())
    transform = T.Compose(transform)
    
     
    if dataset == 'UFMR_DCMR_Supervised':
        dataset = UFMR_DCMR_Supervised(image_dir, attr_path, selected_attrs, transform, mode,image_size,using_ufmr_diffmap)
    
    print(len(dataset))
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=2,
                                  shuffle=(mode=='train_valid'),
                                  num_workers=1)
    return data_loader
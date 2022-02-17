import os.path
from data.base_dataset import BaseDataset, get_transform, get_transform_seg
from data.image_folder import make_dataset, make_labeled_path_dataset, make_dataset_path
from data.online_creation import crop_image, sanitize_paths,write_paths_file
from PIL import Image
import random
import numpy as np
import torchvision.transforms as transforms
import torch
import torchvision.transforms.functional as F

class UnalignedLabeledMaskOnlineDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets with mask labels.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    
    Domain A must have labels, at the moment there are two subdirections 'images' and 'labels'.
    
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        btoA = self.opt.data_direction == 'BtoA'
        self.input_nc = self.opt.model_output_nc if btoA else self.opt.model_input_nc       # get the number of channels of input image
        output_nc = self.opt.model_input_nc if btoA else self.opt.model_output_nc      # get the number of channels of output image
        
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        if os.path.exists(self.dir_A):
            self.A_img_paths, self.A_label_paths = make_labeled_path_dataset(self.dir_A,'/paths.txt')   # load images from '/path/to/data/trainA/paths.txt' as well as labels
            if opt.phase == 'train' and opt.train_compute_D_accuracy:
                self.dir_val_A = os.path.join(opt.dataroot, 'validationA')  # create a path '/path/to/data/trainA'
                self.A_img_paths_val, self.A_label_paths_val = make_labeled_path_dataset(self.dir_val_A,'/paths.txt')   # load images from '/path/to/data/validationA/paths.txt' as well as labels
                
        else:
            self.A_img_paths, self.A_label_paths = make_labeled_path_dataset(opt.dataroot,'/paths.txt')   # load images from '/path/to/data/trainA/paths.txt' as well as labels
        
        if os.path.exists(self.dir_B):
            self.B_img_paths, self.B_label_paths = make_labeled_path_dataset(self.dir_B,'/paths.txt')    # load images from '/path/to/data/trainB'
            if opt.phase == 'train' and opt.train_compute_D_accuracy:
                self.dir_val_B = os.path.join(opt.dataroot, 'validationA')  # create a path '/path/to/data/validationA'
                self.B_img_paths_val, self.B_label_paths_val = make_labeled_path_dataset(self.dir_val_B,'/paths.txt')   # load images from '/path/to/data/validationB/paths.txt' as well as labels

        if self.opt.data_sanitize_paths:
            train_sanitized_exist = os.path.exists(os.path.join(self.sv_dir,'paths_sanitized_train_A.txt')) and os.path.exists(os.path.join(self.sv_dir,'paths_sanitized_train_B.txt'))
            validation_is_needed = self.opt.phase == 'train' and self.opt.train_compute_D_accuracy
            validation_sanitized_exist = os.path.exists(os.path.join(self.sv_dir,'paths_sanitized_validation_B.txt')) and os.path.exists(os.path.join(self.sv_dir,'paths_sanitized_validation_A.txt'))
            if train_sanitized_exist and (not validation_is_needed or validation_sanitized_exist) :
                self.A_img_paths, self.A_label_paths = make_labeled_path_dataset(self.sv_dir,'/paths_sanitized_train_A.txt')
                self.B_img_paths, self.B_label_paths = make_labeled_path_dataset(self.sv_dir,'/paths_sanitized_train_B.txt')
                if validation_is_needed:
                    self.A_img_paths_val, self.A_label_paths_val = make_labeled_path_dataset(self.sv_dir,'/paths_sanitized_validation_A.txt')
                    self.B_img_paths_val, self.B_label_paths_val = make_labeled_path_dataset(self.sv_dir,'/paths_sanitized_validation_B.txt')
                print("Sanitized images and labels paths loaded.")
            else:
                self.sanitize()
        elif opt.data_max_dataset_size!=float("inf"):
            self.A_img_paths, self.A_label_paths=self.A_img_paths[:opt.data_max_dataset_size], self.A_label_paths[:opt.data_max_dataset_size]
            self.B_img_paths, self.B_label_paths=self.B_img_paths[:opt.data_max_dataset_size], self.B_label_paths[:opt.data_max_dataset_size]
            
        self.A_size = len(self.A_img_paths)  # get the size of dataset A
        if os.path.exists(self.dir_B):
            self.B_size = len(self.B_img_paths)  # get the size of dataset B
            
        self.transform=get_transform_seg(self.opt, grayscale=(self.input_nc == 1))
        self.transform_noseg=get_transform(self.opt, grayscale=(self.input_nc == 1))

        self.opt = opt

    def sanitize(self):
        print('--------------')
        print('Sanitizing images and labels paths')
        print('--- DOMAIN A ---')

        self.A_img_paths, self.A_label_paths = sanitize_paths(self.A_img_paths, self.A_label_paths,mask_delta=self.opt.data_online_creation_mask_delta_A,crop_delta=self.opt.data_online_creation_crop_delta_A,mask_square=self.opt.data_online_creation_mask_square_A,crop_dim=self.opt.data_online_creation_crop_size_A,output_dim=self.opt.data_load_size,max_dataset_size=self.opt.data_max_dataset_size)
        write_paths_file(self.A_img_paths, self.A_label_paths,os.path.join(self.sv_dir,'paths_sanitized_train_A.txt'))
        if self.opt.phase == 'train' and self.opt.train_compute_D_accuracy:
            self.A_img_paths_val, self.A_label_paths_val = sanitize_paths(self.A_img_paths_val, self.A_label_paths_val,mask_delta=self.opt.data_online_creation_mask_delta_A,crop_delta=self.opt.data_online_creation_crop_delta_A,mask_square=self.opt.data_online_creation_mask_square_A,crop_dim=self.opt.data_online_creation_crop_size_A,output_dim=self.opt.data_load_size,max_dataset_size=self.opt.train_pool_size)
            write_paths_file(self.A_img_paths_val, self.A_label_paths_val,os.path.join(self.sv_dir,'paths_sanitized_validation_A.txt'))
        print('--- DOMAIN B ---')
        self.B_img_paths, self.B_label_paths = sanitize_paths(self.B_img_paths, self.B_label_paths,mask_delta=self.opt.data_online_creation_mask_delta_B,crop_delta=self.opt.data_online_creation_crop_delta_B,mask_square=self.opt.data_online_creation_mask_square_B,crop_dim=self.opt.data_online_creation_crop_size_B,output_dim=self.opt.data_load_size,max_dataset_size=self.opt.data_max_dataset_size)
        write_paths_file(self.B_img_paths, self.B_label_paths,os.path.join(self.sv_dir,'paths_sanitized_train_B.txt'))
        if self.opt.phase == 'train' and self.opt.train_compute_D_accuracy:
            self.B_img_paths_val, self.B_label_paths_val = sanitize_paths(self.B_img_paths_val, self.B_label_paths_val,mask_delta=self.opt.data_online_creation_mask_delta_B,crop_delta=self.opt.data_online_creation_crop_delta_B,mask_square=self.opt.data_online_creation_mask_square_B,crop_dim=self.opt.data_online_creation_crop_size_B,output_dim=self.opt.data_load_size,max_dataset_size=self.opt.train_pool_size)
            write_paths_file(self.B_img_paths_val, self.B_label_paths_val,os.path.join(self.sv_dir,'paths_sanitized_validation_B.txt'))
        print('--------------')

    def get_img(self,A_img_path,A_label_path,B_img_path=None,B_label_path=None,index=None):
        try:
            A_img,A_label = crop_image(A_img_path,A_label_path,mask_delta=self.opt.data_online_creation_mask_delta_A,crop_delta=self.opt.data_online_creation_crop_delta_A,mask_square=self.opt.data_online_creation_mask_square_A,crop_dim=self.opt.data_online_creation_crop_size_A,output_dim=self.opt.data_load_size)

        except Exception as e:
            print('failure with loading A domain image ', A_img_path, ' or label ', A_label_path)
            print(e)
            return None
       
        A,A_label = self.transform(A_img,A_label)
        if self.opt.f_s_all_classes_as_one:
            A_label = (A_label >= 1)*1

        if B_img_path is not None:
            try:
                if B_label_path is not None:
                    B_img , B_label = crop_image(B_img_path,B_label_path,mask_delta=self.opt.data_online_creation_mask_delta_B,crop_delta=self.opt.data_online_creation_crop_delta_B,mask_square=self.opt.data_online_creation_mask_square_B,crop_dim=self.opt.data_online_creation_crop_size_B,output_dim=self.opt.data_load_size)
                    B,B_label = self.transform(B_img,B_label)
                    if self.opt.f_s_all_classes_as_one:
                        B_label = (B_label >= 1)*1

                else:
                    B_img = Image.open(B_img_path).convert('RGB')
                    B = self.transform_noseg(B_img)
                    B_label = []
                
            except Exception as e:
                print("failure with loading B domain image ", B_img_path, 'or label', B_label_path)
                print(e)
                return None

            return {'A': A, 'B': B, 'A_img_paths': A_img_path, 'B_img_paths': B_img_path, 'A_label': A_label, 'B_label': B_label}
        else:
            return {'A': A, 'A_img_paths': A_img_path,'A_label': A_label}
                
    def __len__(self):
        """Return the total number of images in the dataset.
        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        if hasattr(self,'B_img_paths'):
            return max(self.A_size, self.B_size)
        else:
            return self.A_size
import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from skimage.filters import frangi, hessian
from skimage.transform import resize

from pl_bolts.models.self_supervised.simclr.transforms import SimCLRTrainDataTransform, SimCLREvalDataTransform, SimCLRFinetuneTransform


dl_info_path = './data/DL_info.csv'
subset_map = {'train': 1, 'valid': 2, 'test': 3}
SKIP = [660, 6679, 6680, 9943, 11078, 11338, 18692, 19957, 22403]   # Skip non-accessible images

vae_transforms = [
    transforms.ToTensor(), 
    transforms.Compose([transforms.RandomHorizontalFlip(p=1), transforms.ToTensor()]), 
    transforms.Compose([transforms.RandomVerticalFlip(p=1), transforms.ToTensor()]), 
#     transforms.Compose([transforms.RandomRotation((90, 90)), transforms.ToTensor()])
]

normalize_args = {
    'frangi': transforms.Normalize([0.4520], [0.4965]),
    'hessian': transforms.Normalize([0.3817], [0.4858]),
    'none': transforms.Normalize([0.4985], [0.0059]),
}

def get_test_loader(image_dim, batch_size, use_filter):
    test_loader = DataLoader(LesionDataset('test', use_filter=use_filter), shuffle=False, batch_size=batch_size, num_workers=8)

    return test_loader

def get_pretrain_loaders(model='vae', dim=64, batch_size=16, use_filter=True):
    '''Get pretrain data loaders'''

    if model == 'vae':
        vae_transforms.append(normalize_args[use_filter])
        
        train_dataset = ConcatDataset([LesionDataset('train', use_filter=use_filter, transform=i) for i in vae_transforms])
        valid_dataset = LesionDataset('valid', use_filter=use_filter)
    
    elif model == 'simclr':
        simclr_train_transform = SimCLRTrainDataTransform(
            input_height=dim,
            gaussian_blur=False,
            jitter_strength=0.5,
            normalize=normalize_args[use_filter])

        simclr_eval_transform = SimCLREvalDataTransform(
            input_height=dim,
            gaussian_blur=False,
            jitter_strength=0.5,
            normalize=normalize_args[use_filter])
        
        train_dataset = LesionDataset('train', use_filter=use_filter, transform=simclr_train_transform)
        valid_dataset = LesionDataset('valid', use_filter=use_filter, transform=simclr_eval_transform)
    
    else:
        raise NotImplementedError

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=8)
    valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=batch_size, num_workers=8)
    test_loader = DataLoader(LesionDataset('test', use_filter=use_filter), shuffle=False, batch_size=batch_size, num_workers=8)

    return train_loader, valid_loader, test_loader, len(train_dataset)

def get_classification_loaders(model='vae', dim=64, batch_size=16, use_filter=True):
    '''Get lesion type classification data loaders'''

    if model == 'vae':
        vae_transforms.append(normalize_args[use_filter])
        
        train_dataset = ConcatDataset([LesionDataset('valid', use_filter=use_filter, transform=i) for i in vae_transforms])
        valid_dataset = LesionDataset('test', use_filter=use_filter)

    elif model == 'simclr':
        simclr_train_transform = SimCLRFinetuneTransform(
            input_height=dim,
            normalize=normalize_args[use_filter],
            eval_transform=False)

        simclr_eval_transform = SimCLRFinetuneTransform(
            input_height=dim,
            normalize=normalize_args[use_filter],
            eval_transform=True)
            
        train_dataset = LesionDataset('valid', use_filter=use_filter, transform=simclr_train_transform)
        valid_dataset = LesionDataset('test', use_filter=use_filter, transform=simclr_eval_transform)

    else:
        raise NotImplementedError

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=8, drop_last=True)
    valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=batch_size, num_workers=8, drop_last=True)

    return train_loader, valid_loader, valid_dataset.num_labels

def get_cbir_loader(train_batch_size, eval_batch_size):
    train_dataset = CBIRTrainDataset('./data/cbir_data.csv', neg_size=1, use_filter='frangi')
    valid_dataset = CBIREvalDataset(use_filter='frangi')

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=4)

    return train_loader, valid_loader

def get_federated_loaders(num_clients, dim=64, batch_size=8, use_filter='frangi'):

    df = pd.read_csv(dl_info_path)
    federated_train_loaders = []
    local_data_size =  1 / num_clients
    for i in range(num_clients):
        local_df= train_test_split(df, stratify=df.Coarse_lesion_type, test_size=local_data_size)
        local_dataset = LesionDataset(
            'valid', 
            use_filter=use_filter, 
            transform=SimCLRFinetuneTransform(input_height=dim, eval_transform=False))
        local_loader = DataLoader(local_dataset, shuffle=True, batch_size=batch_size, num_workers=8)
        federated_train_loaders.append(local_loader)

    test_dataset = LesionDataset('test', use_filter=use_filter, transform=SimCLRFinetuneTransform(input_height=dim, eval_transform=True))
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, num_workers=8)

    return federated_train_loaders, test_loader


class LesionDataset(Dataset):
    
    def __init__(self, subset, num_slice=1, lesion_types=None, 
                 dl_info=dl_info_path, 
                 root='./images', 
                 transform=None, 
                 use_filter=False):        
        self.root = root
        self.num_slice = num_slice

        if subset != 'all':
            self.subset = [subset_map[subset]]
        else:
            self.subset = list(subset_map.values())
        
        if type(dl_info) == str:
            df = pd.read_csv(dl_info)
        else:
            df = dl_info
        
        if lesion_types is not None:
            df = df[df['Coarse_lesion_type'].isin(lesion_types)]
        
        df = df[~df.index.isin(SKIP)]
        self.df = df[df.Train_Val_Test.isin(self.subset)].reset_index(drop=True)
        
        self.transform = transform
        self.use_filter = use_filter

        self.df['label'] = self.df['Coarse_lesion_type'] - 1
        self.num_labels = len(self.df.label.unique())
    
    def __len__(self):
        return self.df.shape[0]
    
    def crop(self, img, bbox):
        FIXED_SIZE = (64,64)

        bbox = bbox.split(',')
        bbox = [int(float(v)) for v in bbox]

        # convert bbox to fixed size
        diff_x = bbox[2] - bbox[0]
        diff_y = bbox[3] - bbox[1]

        pad_x = FIXED_SIZE[0] - diff_x
        pad_y = FIXED_SIZE[1] - diff_y

        # Take care of out-of-bound index
        x_min = 0
        x_max = img.shape[0]
        y_min = 0
        y_max = img.shape[1]

        x_start = bbox[1]-int(pad_y/2)
        if x_start < x_min:
            x_start = x_min
        x_end = bbox[3]+int(pad_y/2)

        if x_end > x_max:
            x_end = x_max

        y_start = bbox[0]-int(pad_x/2)
        if y_start < y_min:
            y_start = y_min

        y_end = bbox[2]+int(pad_x/2)
        if y_end > y_max:
            y_end = y_max

        padded_lesion = img[x_start:x_end, y_start:y_end]
        padded_lesion = resize(padded_lesion, FIXED_SIZE)

        return padded_lesion
    
    def normalize_image(self, img):
        norm_func = normalize_args[self.use_filter]

        return norm_func(img)
        
    def get_image(self, bbox, file_name):
        folder = '_'.join(file_name.split('_')[:-1])
        img_file = file_name.split('_')[-1]
        img = cv2.imread(os.path.join(self.root, folder, img_file))
        img = img[:, :, 1] / 255

        img = self.crop(img, bbox)
        
        # Use filter to create more contrastive values
        if self.use_filter == 'none':
            pass
        elif self.use_filter == 'frangi':
            img = frangi(img, sigmas=(2,3), scale_step=1, beta=5, gamma=0.00000002, black_ridges=True)
        elif self.use_filter == 'hessian':
            img = hessian(img, mode='reflect')
            # img[img < 0.5] = 0
            # img[img > 0.5] = 1
            # img[img < 0] = 0
        else:
            raise NotImplementedError

        # Convert to tensor for transformation
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)
        elif len(img.shape) == 3:
            img = img.transpose((-1, 0, 1))
        else:
            raise NotImplementedError
        img = torch.FloatTensor(img.astype(np.float32))

        # Data augmentation
        if self.transform:
            img = transforms.ToPILImage()(img)
            img = self.transform(img)

        return img
    
    def __getitem__(self, idx):
            
        label = self.df.iloc[idx]['label']
        bbox = self.df.iloc[idx]['Bounding_boxes']
        file_name = self.df.iloc[idx]['File_name']
        
        img = self.get_image(bbox, file_name)

        return img, label

class CBIRTrainDataset(LesionDataset):
    
    def __init__(self, data, neg_size=16, transform=None, use_filter='none'):
        df = pd.read_csv(dl_info_path)
        self.df = df[df.Train_Val_Test != subset_map['train']].reset_index(drop=True)
        self.df['label'] = self.df['Coarse_lesion_type'] - 1
        
        data = pd.read_csv(data)
        data['query'] = data['query'].apply(eval)
        data['pos'] = data['pos'].apply(eval)
        data['neg'] = data['neg'].apply(eval)
        self.data = data

        self.root = './images'
        self.transform = transform
        self.use_filter = use_filter
        self.neg_size = neg_size
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data.iloc[idx]
        query, pos, neg = data['query'], data['pos'], data['neg']
        # pos_head, pos_tail, neg_tails = self.df.iloc[data['query']], self.df.iloc[data['pos']], self.df.iloc[data['neg']]
        # pos_head = [pos_head['Bounding_boxes'], pos_head['File_name']]
        # pos_tail = [pos_tail['Bounding_boxes'], pos_tail['File_name']]
        # neg_tails = [(i, j) for i, j in zip(neg_tails['Bounding_boxes'], neg_tails['File_name'])]

        query_img = self.get_image(*query[2:])
        pos_img = self.get_image(*pos[2:])

        if self.neg_size > 1:
            neg_img = []
            for t in range(self.neg_size):
                
                img = self.get_image(*neg[t])
                neg_img.append(img)
            
            neg_img = torch.vstack(neg_img)
        else:
            neg_img = self.get_image(*neg[0][2:])

        return query_img, pos_img, neg_img

class CBIREvalDataset(LesionDataset):
    
    def __init__(self, transform=None, use_filter='none', patient='all'):
        df = pd.read_csv(dl_info_path)
        self.df = df[df.Train_Val_Test == subset_map['test']].reset_index(drop=True)
        self.df['label'] = self.df['Coarse_lesion_type'] - 1

        self.root = './images'
        self.transform = transform
        self.use_filter = use_filter
        self.patient = patient

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        data = self.df.iloc[idx]

        patient_id = data['Patient_index']
        label = data['label']
        img = self.get_image(data['Bounding_boxes'], data['File_name'])

        all_img_idx = list(self.df.index)
        same_img_idx = list(self.df[self.df.Patient_index == patient_id].index)
        diff_img_idx = list(self.df[self.df.Patient_index != patient_id].index)

        all_mask = np.zeros((self.df.shape[0],)).astype(bool)
        all_mask[all_img_idx] = True

        same_mask = np.zeros((self.df.shape[0],)).astype(bool)
        same_mask[same_img_idx] = True

        diff_mask = np.zeros((self.df.shape[0],)).astype(bool)
        diff_mask[diff_img_idx] = True
        
        return img, label, all_mask, same_mask, diff_mask


'''
https://discuss.pytorch.org/t/error-with-functional-image-rotation/73366/6
https://discuss.pytorch.org/t/error-while-multiprocessing-in-dataloader/46845/9
'''
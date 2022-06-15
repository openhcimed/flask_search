import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from skimage.filters import frangi, hessian
from skimage.transform import resize


dl_info_path = './DL_info.csv'
subset_map = {'train': 1, 'valid': 2, 'test': 3}
SKIP = [660, 6679, 6680, 9943, 11078, 11338, 18692, 19957, 22403]   # Skip non-accessible images

def func(x):
    img_path = [i for i in zip(x.Patient_index, x.Coarse_lesion_type, x.Bounding_boxes, x.File_name)]
    
    return img_path

def prepare_retrieval_dataset(neg_size=16):
    """Prepare a retrieval dataset for ranking system"""

    df = pd.read_csv(dl_info_path)
    df = df[df.Train_Val_Test == subset_map['valid']].reset_index(drop=True)
    df['label'] = df['Coarse_lesion_type'] - 1

    all_pairs = df.groupby('label').apply(func).to_dict()
    data = []
    all_labels = list(all_pairs.keys())

    for pos_label, items in all_pairs.items():
        
        # Randomly select 100 pairs to construct dataset
        # It's too big to construct all possible pairs
        indices = np.random.choice(len(items), 100, replace=False)
        pos_items = [items[i] for i in indices]
        
        for head in pos_items:
            for pos_tail in pos_items:
                
                if head != pos_tail:

                    # Randomly select negative pairs of other labels
                    all_neg_tails = []
                    for i in range(neg_size):

                        # Select a negative label
                        neg_labels = list(all_labels)
                        neg_labels.remove(pos_label)
                        random_neg_label = np.random.choice(neg_labels)

                        # Select a negative pair
                        neg_candidates = all_pairs[random_neg_label]
                        neg_idx = np.random.choice(len(neg_candidates))
                        neg_tail = neg_candidates[neg_idx]
                        all_neg_tails.append(neg_tail)
                    
                    data.append({
                        'query': head, 
                        'pos': pos_tail, 
                        'neg': all_neg_tails
                    })
    
    np.random.shuffle(data)
    train_split = int(len(data) * 0.6)
    valid_split = int(len(data) * 0.8)

    data = pd.DataFrame(data)
    data.loc[:train_split, 'split'] = 'train'
    data.loc[train_split:valid_split, 'split'] = 'valid'
    data.loc[valid_split:, 'split'] = 'test'
    
    data.to_csv('./cbir_data.csv', index=None)


prepare_retrieval_dataset()

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')

import os
import glob
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from lesion_image_learning.models import LesionResnetSimCLR, LesionFinetuner
from pl_bolts.models.self_supervised.ssl_finetuner import SSLFineTuner

from lesion_image_learning.dataset import get_loader_csl_pretrain, get_loader_csl_finetune
from lesion_image_learning.utils import *

from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import euclidean_distances


torch.autograd.set_detect_anomaly(True)

def parse_args():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('--latent_dim', type=int, default=128, help='latent dimension of VAE')

    # Data
    parser.add_argument('--image_dim', type=int, default=120, 
                        help='dimension of lesion image')
    parser.add_argument('--use_filter', action='store_true', default=False,
                        help='use filter to preprocess images')
    parser.add_argument('--use_augmentation', action='store_true', default=False,
                        help='use data augmentation')  
    parser.add_argument('--random_seed', type=int, default=1234, help='random seed')

    # Training
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer')
    
    # Checkpoint
    parser.add_argument('--pretrained_model_pretrain_only', type=str, 
                        default='./lightning_logs/lesion_simclr_resnet18_128/',
                        # default='./lightning_logs/version_19884525/checkpoints/epoch=49-step=15299.ckpt', 
                        help='path to the pretrained SimCLR model')
    parser.add_argument('--pretrained_model_finetune_only', type=str, 
                        default='./lightning_logs/simclr_finetune_only/',
                        help='path to the finetuned SimCLR model')
    parser.add_argument('--pretrained_model_finetune_pretrain', type=str, 
                        default='./lightning_logs/simclr_pretrain_finetune/',
                        help='path to the finetuned SimCLR model')

    args = parser.parse_args()

    return args

def get_checkpoint(path):
    return glob.glob(f'{path}/checkpoints/*.ckpt')[0]

def main():
    args = parse_args()
    pl.seed_everything(args.random_seed)

    # Dataset
    _, _, test_loader, _ = get_loader_csl_pretrain(args.image_dim, args.batch_size, args.use_filter)
    _, _, num_labels = get_loader_csl_finetune(args.image_dim, args.batch_size, args.use_filter)

    # Model and optimizer
    resnet_out_dim = {'resnet18': 512, 'resnet50': 2048}
    for pretrained_model, name in zip(
        [args.pretrained_model_pretrain_only, args.pretrained_model_finetune_only, args.pretrained_model_finetune_pretrain],
        ['pretrain_only', 'finetune_only', 'finetune_pretrain']):

        print(name)

        if name == 'pretrain_only':
            continue
            model = LesionResnetSimCLR.load_from_checkpoint(
                get_checkpoint(pretrained_model),
                input_channel=1, 
                arch='resnet18', 
                hidden_mlp=resnet_out_dim['resnet18'], 
                feat_dim=args.latent_dim, 
                batch_size=args.batch_size, 
                learning_rate=args.lr, 
                dataset='lesion', 
                num_samples=1, 
                gpus=1)
        
        else:
            model = LesionResnetSimCLR(
                input_channel=1, 
                arch='resnet18', 
                hidden_mlp=resnet_out_dim['resnet18'], 
                feat_dim=args.latent_dim, 
                batch_size=args.batch_size, 
                learning_rate=args.lr, 
                dataset='lesion', 
                num_samples=1, 
                gpus=1)
            
            finetuner = LesionFinetuner.load_from_checkpoint(
                get_checkpoint(pretrained_model),
                backbone=model,
                in_features=resnet_out_dim['resnet18'],
                num_classes=num_labels,
                hidden_dim=None,
                epochs=args.epochs,
                learning_rate=args.lr,
                dropout=args.dropout
            )
            model = finetuner.backbone
        
        model.cuda()
        
        # Get vectors
        raw_vectors = []
        vectors = []
        labels = []
        for images, types in tqdm(test_loader, desc=name):
            with torch.no_grad():
                v = model(images.to('cuda'))
            raw_vectors.extend(images.reshape((images.shape[0], -1)))
            vectors.extend(v.cpu().detach().numpy())
            labels.extend(types.cpu().detach().numpy())

        vectors = np.vstack(vectors)
        labels = np.vstack(labels).squeeze()

        import pickle
        with open(f'./raw_vectors.pkl', 'wb') as f:
            pickle.dump(raw_vectors, f)
        with open(f'./vectors_{name}.pkl', 'wb') as f:
            pickle.dump(vectors, f)
        with open(f'./labels_{name}.pkl', 'wb') as f:
            pickle.dump(labels, f)
    
    # for idx, (query_vector, query_label) in enumerate(zip(vectors, labels)):
        
    #     pred = euclidean_distances(query_vector, vectors)

    #     true_mask = labels == query_label
    #     true_mask[idx] = False
    #     true_label = vectors[true_mask]

    # print(len(test_loader), vectors.shape, labels.shape)


if __name__ == "__main__":
    main()

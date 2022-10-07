import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import os
import glob
import copy
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pl_bolts.callbacks import PrintTableMetricsCallback

from deep_lesion.data_loader import get_cbir_loader
from deep_lesion.models.vae import LesionVAE
from deep_lesion.models.contrastive import LesionSimCLR
from deep_lesion.models.cbir import SimCLRRMAC
from deep_lesion.utils import *


def parse_args():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('--model', type=str, default='simclr', help='model name')
    parser.add_argument('--mode', type=str, default='vec_dis', help='training mode')
    parser.add_argument('--latent_dim', type=int, default=256, help='latent dimension of VAE')

    # Data
    parser.add_argument('--image_dim', type=int, default=64, 
                        help='dimension of lesion image')
    parser.add_argument('--use_filter', type=str, default='frangi',
                        help='use filter to preprocess images')
    parser.add_argument('--use_augmentation', action='store_true', default=False,
                        help='use data augmentation')  
    parser.add_argument('--random_seed', type=int, default=1234, help='random seed')

    # Training
    parser.add_argument('--margin', type=int, default=1, help='margin of triplet loss')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--train_batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--eval_batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--negative_sample_size', type=int, default=16, help='number of negative pairs per positive pair')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer')
    parser.add_argument('--scheduler_type', type=str, default='cosine', help='LR scheduler')
    
    # Checkpoint
    parser.add_argument('--init_checkpoint', action='store_true', default=False, 
                        help='whether to start from checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', 
                        help='path to save/load checkpoint')
    parser.add_argument('--pretrained_model', type=str, 
                        default='pretrain-simclr-64-256-frangi', 
                        help='path to the pretrained SimCLR model')

    args = parser.parse_args()

    if args.model not in ['vae', 'simclr']:
        raise NotImplementedError

    # if args.model == 'vae' and torch.cuda.device_count() > 1:
    #     args.batch_size = args.batch_size * torch.cuda.device_count()

    return args

def precision_at_k(pred, k=50):
    
    return sum(pred[:k]) / k

def mean_average_precision(pred, k=50):
    
    return np.mean([precision_at_k(pred, i) for i in range(1, k+1)])

def rank(query, candidates, model, embed_candidates=True):

    with torch.no_grad():
        query_emb = model(query).cpu()
    
    if embed_candidates:
        with torch.no_grad():
            candi_emb = model(candidates).cpu()
    else:
        candi_emb = candidates

    query_emb = F.normalize(query_emb).numpy()
    candi_emb = F.normalize(candi_emb).numpy()
    
    # query_emb = query_emb.numpy()
    # candi_emb = candi_emb.numpy()
    
    distances = np.linalg.norm(query_emb - candi_emb, axis=-1)
    ranking = np.argsort(distances)

    return distances, ranking

def get_rep(model, valid_loader):
    all_rep = []
    all_label = []
    for img, label, all_mask, same_mask, diff_mask in tqdm(valid_loader):
        with torch.no_grad():
            rep = model(img.to('cuda'))
            all_rep.append(rep.cpu())
            all_label.append(label)

    all_rep = torch.cat(all_rep)
    all_label = torch.cat(all_label)

    return all_rep, all_label

def eval_cbir(model, valid_loader, all_rep, all_label, mask_type):
    total_ap = 0
    total_p1 = 0
    total_p3 = 0
    total_p5 = 0
    total_p10 = 0
    for data in tqdm(valid_loader):
        query, query_label, all_mask, same_mask, diff_mask = data
        
        if mask_type == 'all':
            mask = all_mask
        
        elif mask_type == 'same':
            mask = same_mask

        elif mask_type == 'diff':
            mask = diff_mask
        
        candidates = all_rep[mask.squeeze()]
        labels = all_label[mask.squeeze()]

        distance, indices = rank(query.to('cuda'), candidates, model, embed_candidates=False)
        binary_labels = (labels == query_label).int().numpy()

        # Sort labels based on distances
        pred = binary_labels[indices]

        # Metrics
        ap = mean_average_precision(pred, k=10)  # 1st is query itself
        precision_at_1 = precision_at_k(pred, k=1)
        precision_at_3 = precision_at_k(pred, k=3)
        precision_at_5 = precision_at_k(pred, k=5)
        precision_at_10 = precision_at_k(pred, k=10)
        precision_at_50 = precision_at_k(pred, k=50)
        precision_at_100 = precision_at_k(pred, k=100)
        precision_at_500 = precision_at_k(pred, k=500)

        total_ap += ap
        total_p1 += precision_at_1
        total_p3 += precision_at_3
        total_p5 += precision_at_5
        total_p10 += precision_at_10

    length = len(valid_loader)
    print(mask_type, total_ap / length, total_p1 / length, total_p3 / length, total_p5 / length, total_p10 / length)

def main():
    # torch.backends.cudnn.enabled = False
    args = parse_args()
    pl.seed_everything(args.random_seed)

    train_loader, valid_loader = get_cbir_loader(args.train_batch_size, 1)

    ckpt_name = f'cbir-{args.model}-{args.image_dim}-{args.latent_dim}-{args.use_filter}'
    if args.model == 'vae' and args.use_augmentation:
        ckpt_name += '-aug'
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.checkpoint_dir, ckpt_name),
        filename='{epoch}-{train_loss:.4f}-{val_loss:.4f}-{val_acc:.4f}',
        monitor='val_loss'
    )

    print('Training config', ckpt_name)
    
    resnet_out_dim = {'resnet18': 512, 'resnet50': 2048}

    # if args.pretrained_model is not None:
    #     pretrained_model = os.path.join('lightning_logs', 'version_22022152', 'checkpoints')
    #     pretrained_model = glob.glob(f'{pretrained_model}/*ckpt')[0]
    
    if args.model == 'vae':
        encoder = LesionVAE.load_from_checkpoint(
            args.pretrained_model, 
            input_channel=1, 
            arch='resnet18', 
            hidden_mlp=resnet_out_dim['resnet18'], 
            feat_dim=args.latent_dim, 
            batch_size=args.train_batch_size, 
            learning_rate=args.lr, 
            dataset='lesion', 
            num_samples=1, 
            gpus=1)
        
        model = LesionVAEForRetrieval(
            backbone=encoder,
            in_features=resnet_out_dim['resnet18'],
            hidden_dim=args.latent_dim,
            epochs=args.epochs,
            learning_rate=args.lr,
            dropout=args.dropout,
            image_dim=args.image_dim, 
            latent_dim=args.latent_dim, 
            use_filter=args.use_filter)
    
    elif args.model == 'simclr':
        encoder = LesionSimCLR.load_from_checkpoint(
            args.pretrained_model,
            input_channel=1, 
            arch='resnet18', 
            hidden_mlp=resnet_out_dim['resnet18'], 
            feat_dim=args.latent_dim, 
            batch_size=args.train_batch_size, 
            learning_rate=args.lr, 
            dataset='lesion', 
            num_samples=1, 
            gpus=1)

        model = SimCLRRMAC(
            backbone=encoder,
            epochs=args.epochs,
            image_dim=args.image_dim, 
            latent_dim=args.latent_dim, 
            use_filter=args.use_filter, 
            dropout=args.dropout,
            margin=args.margin,
            lr=args.lr,
            weight_decay=args.weight_decay,
            scheduler_type=args.scheduler_type,
            mode='vec_dis')

    else:
        raise NotImplementedError

    wandb_logger = WandbLogger(name=ckpt_name, offline=True)

    # Trainer
    trainer = pl.Trainer(
        gpus=1, 
        max_epochs=args.epochs, 
        logger=wandb_logger,
        callbacks=[PrintTableMetricsCallback(), checkpoint_callback])

    
    # Train
    trainer.fit(model, train_dataloaders=train_loader)

    # Evaluation
    model = model.to('cuda')
    all_rep, all_label = get_rep(model, valid_loader)

    for mask_type in ['all', 'same', 'diff']:
        eval_cbir(model, valid_loader, all_rep, all_label, mask_type)

if __name__ == "__main__":
    main()

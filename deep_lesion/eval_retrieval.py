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
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pl_bolts.callbacks import PrintTableMetricsCallback

from deep_lesion.data_loader import RetrievalDataset, get_cbir_eval_loader
from deep_lesion.models.vae import LesionVAE
from deep_lesion.models.contrastive import LesionSimCLR, LesionSimCLRForRetrival
from deep_lesion.models.cbir import SimCLRRMAC
from deep_lesion.utils import *


def parse_args():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('--model', type=str, default='simclr', help='model name')
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
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--train_batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--eval_batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--negative_sample_size', type=int, default=16, help='number of negative pairs per positive pair')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer')
    
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

def main():
    # torch.backends.cudnn.enabled = False
    args = parse_args()
    pl.seed_everything(args.random_seed)

    # data = pd.read_csv('./data/cbir_data.csv')
    # data['anchor'] = data['anchor'].apply(eval)
    # data['pos'] = data['pos'].apply(eval)
    # data['neg'] = data['neg'].apply(eval)

    # if args.negative_sample_size == 1:
    #     data['neg'] = data['neg'].apply(lambda x: x[0])

    all_loader, same_patient_loader, diff_patient_loader = get_cbir_eval_loader(args.eval_batch_size)
    
    ckpt_name = f'finetune-{args.model}-{args.image_dim}-{args.latent_dim}-{args.use_filter}'
    if args.model == 'vae' and args.use_augmentation:
        ckpt_name += '-aug'
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.checkpoint_dir, ckpt_name), 
        monitor='val_loss')

    print('Training config', ckpt_name)
    
    resnet_out_dim = {'resnet18': 512, 'resnet50': 2048}

    if args.pretrained_model is not None:
        pretrained_model = os.path.join('lightning_logs', args.pretrained_model, 'checkpoints')
        pretrained_model = glob.glob(f'{pretrained_model}/*ckpt')[0]
    
    if args.model == 'vae':
        encoder = LesionVAE.load_from_checkpoint(
            pretrained_model, 
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
            pretrained_model,
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
            in_features=resnet_out_dim['resnet18'],
            hidden_dim=args.latent_dim,
            dropout=args.dropout,
            image_dim=args.image_dim, 
            latent_dim=args.latent_dim, 
            use_filter=args.use_filter, 
            neg_size=args.negative_sample_size)

    else:
        raise NotImplementedError

    # Trainer
    trainer = pl.Trainer(
        accelerator='ddp',
        gpus=torch.cuda.device_count(),
        max_epochs=args.epochs, 
        callbacks=[PrintTableMetricsCallback()])
    
    # Test
    trainer.test(model, test_dataloaders=all_loader)
    trainer.test(model, test_dataloaders=same_patient_loader)
    trainer.test(model, test_dataloaders=diff_patient_loader)

if __name__ == "__main__":
    main()

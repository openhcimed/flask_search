import os
import glob
import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pl_bolts.callbacks import PrintTableMetricsCallback

from deep_lesion.data_loader import get_classification_loaders
from deep_lesion.models.vae import LesionVAE, LesionVAEForFinetune
from deep_lesion.models.contrastive import LesionSimCLR, LesionSimCLRForFinetune
from deep_lesion.utils import *


torch.autograd.set_detect_anomaly(True)

def parse_args():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('--model', type=str, default='simclr', help='model name')
    parser.add_argument('--latent_dim', type=int, default=256, help='latent dimension of VAE')

    # Data
    parser.add_argument('--image_dim', type=int, default=64, 
                        help='dimension of lesion image')
    parser.add_argument('--use_filter', type=str, default='none',
                        help='use filter to preprocess images')
    parser.add_argument('--use_augmentation', action='store_true', default=True,
                        help='use data augmentation')  
    parser.add_argument('--random_seed', type=int, default=1234, help='random seed')

    # Training
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
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
                        default=None, 
                        help='path to the pretrained model')

    args = parser.parse_args()

    if args.model not in ['vae', 'simclr']:
        raise NotImplementedError

    if args.model == 'vae' and torch.cuda.device_count() > 1:
        args.batch_size = args.batch_size * torch.cuda.device_count()

    return args

def main():
    args = parse_args()
    pl.seed_everything(args.random_seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Dataset
    train_loader, valid_loader, num_labels = get_classification_loaders(
        args.model, args.image_dim, args.batch_size, args.use_filter)

    # Checkpoint
    ckpt_name = f'finetune-{args.model}-{args.image_dim}-{args.latent_dim}-{args.use_filter}'
    if args.model == 'vae' and args.use_augmentation:
        ckpt_name += '-aug'
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.checkpoint_dir, ckpt_name),
        filename='{epoch}-{train_loss:.4f}-{val_loss:.4f}-{val_acc:.4f}',
        monitor=None        # Save last
    )

    print(f'****************** num_labels {num_labels} ******************')
    print('Training config', ckpt_name)

    # Model and optimizer
    resnet_out_dim = {'resnet18': 512, 'resnet50': 2048}

    if args.pretrained_model is not None:
        # pretrained_model = os.path.join('./checkpoint', args.pretrained_model, 'checkpoints')
        # pretrained_model = glob.glob(f'{args.pretrained_model}/*ckpt')[0]
    
        if args.model == 'vae':
            encoder = LesionVAE.load_from_checkpoint(
                args.pretrained_model, 
                input_channel=1, 
                arch='resnet18', 
                hidden_mlp=resnet_out_dim['resnet18'], 
                feat_dim=args.latent_dim, 
                batch_size=args.batch_size, 
                learning_rate=args.lr, 
                dataset='lesion', 
                num_samples=1, 
                gpus=1)
        
        elif args.model == 'simclr':
            encoder = LesionSimCLR.load_from_checkpoint(
                args.pretrained_model,
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
        pretrained_model = None

        if args.model == 'vae':
            encoder = LesionVAE( 
                input_channel=1, 
                arch='resnet18', 
                hidden_mlp=resnet_out_dim['resnet18'], 
                feat_dim=args.latent_dim, 
                batch_size=args.batch_size, 
                learning_rate=args.lr, 
                dataset='lesion', 
                num_samples=1, 
                gpus=1)
        
        elif args.model == 'simclr':
            encoder = LesionSimCLR(
                input_channel=1, 
                arch='resnet18', 
                hidden_mlp=resnet_out_dim['resnet18'], 
                feat_dim=args.latent_dim, 
                batch_size=args.batch_size, 
                learning_rate=args.lr, 
                dataset='lesion', 
                num_samples=1, 
                gpus=1)

    if args.model == 'vae':    
        model = LesionVAEForFinetune(
            num_labels=num_labels,
            backbone=encoder,
            in_features=resnet_out_dim['resnet18'],
            num_classes=num_labels,
            hidden_dim=args.latent_dim,
            epochs=args.epochs,
            learning_rate=args.lr,
            dropout=args.dropout,
            image_dim=args.image_dim, 
            latent_dim=args.latent_dim, 
            use_filter=args.use_filter)
    
    elif args.model == 'simclr':
        model = LesionSimCLRForFinetune(
            num_labels=num_labels,
            backbone=encoder,
            in_features=resnet_out_dim['resnet18'],
            num_classes=num_labels,
            hidden_dim=args.latent_dim,
            epochs=args.epochs,
            learning_rate=args.lr,
            dropout=args.dropout,
            image_dim=args.image_dim, 
            latent_dim=args.latent_dim, 
            use_filter=args.use_filter)        

    
    wandb_logger = WandbLogger(name=ckpt_name, offline=True)

    # Trainer
    trainer = pl.Trainer(
        gpus=1, 
        max_epochs=args.epochs, 
        logger=wandb_logger,
        callbacks=[PrintTableMetricsCallback(), checkpoint_callback])

    # Train
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    trainer.validate(model, dataloaders=valid_loader, ckpt_path='best')

    # trainer.save_checkpoint(os.path.join(args.checkpoint_dir, ckpt_name+'.ckpt'))

if __name__ == "__main__":
    main()

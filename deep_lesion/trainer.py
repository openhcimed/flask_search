from tqdm.notebook import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.models.self_supervised.ssl_finetuner import SSLFineTuner

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LesionFinetuner(SSLFineTuner):
    def __init__(self, **finetuner_kw):
        super().__init__(**finetuner_kw)
        self.linear_layer = nn.Linear(finetuner_kw['hidden_dim'], finetuner_kw['num_classes'])
        # self.linear_layer = nn.Linear(finetuner_kw['in_features'], finetuner_kw['num_classes'])

    def shared_step(self, batch):
        x, y = batch

        # with torch.no_grad(): # This is the difference from the original implementation
        feats = self.backbone(x)

        # feats = self.backbone.get_latent_vector(x)

        feats = feats.view(feats.size(0), -1)
        logits = self.linear_layer(feats)
        loss = F.cross_entropy(logits, y)

        return loss, logits, y

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            list(self.backbone.parameters()) + list(self.linear_layer.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # set scheduler
        if self.scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.decay_epochs, gamma=self.gamma)
        elif self.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                self.epochs,
                eta_min=self.final_lr  # total epochs to run
            )

        return [optimizer], [scheduler]

def vae_loss(recons, inputs, mu, log_var, **kwargs):
    r"""Computes the VAE loss function. 
        KL(N(\mu, \sigma), N(0, 1)) = 
        \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
    """
    batch_size = recons.shape[0]
    recons_loss = F.binary_cross_entropy(recons.flatten(), inputs.flatten(), reduction='sum') / batch_size
    KLD = torch.mean(
        -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1))
    loss = recons_loss + KLD
    return loss, recons_loss, KLD

def do_train(train_loader, model, optimizer, epoch, train_log):
    train_log['loss'].append([])
    train_log['recons_loss'].append([])
    train_log['kl_div'].append([])
    for images, bbox, scale, imgfile in tqdm(train_loader, desc='Train Iteration', leave=False):
        optimizer.zero_grad()
        model.train()

        z, recon_images, mu, log_var = model(images.to(device))
        loss, recons_loss, kl_div = model.module.loss_fn(recon_images, images.to(device), mu, log_var)
        
        if torch.isnan(loss) or torch.isnan(recons_loss) or torch.isnan(kl_div):
            print(loss.item(), recons_loss.item(), kl_div.item())
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        
        train_log['loss'][epoch].append(loss.item())
        train_log['recons_loss'][epoch].append(recons_loss.item())
        train_log['kl_div'][epoch].append(kl_div.item())
    
    train_log['loss'][epoch] = np.array(train_log['loss'][epoch])
    train_log['recons_loss'][epoch] = np.array(train_log['recons_loss'][epoch])
    train_log['kl_div'][epoch] = np.array(train_log['kl_div'][epoch])
    torch.cuda.empty_cache()
    
#     return images, recon_images

def do_eval(eval_loader, model, n_gpu=1, use_valid=True):
    """Validate the model on valid / test set"""
    
    epoch_log = {'loss': [], 'recons_loss': [], 'kl_div': []}
    for images, bbox, scale, imgfile, label in tqdm(eval_loader, desc='Valid', leave=False):
        model.eval()
        z, recon_images, mu, logvar, loss, recons_loss, kl_div = model(images.to(device))
        if n_gpu > 1:
            loss = loss.mean(); recons_loss = recons_loss.mean(); kl_div = kl_div.mean()
        
        epoch_log['loss'].append(loss.item())
        epoch_log['recons_loss'].append(recons_loss.item())
        epoch_log['kl_div'].append(kl_div.item())
        torch.cuda.empty_cache()
       
    result = {}
    for k, v in epoch_log.items():
        result[k] = np.mean(v)
    
    return result

def do_eval_ae(eval_loader, model, n_gpu=1, use_valid=True):
    """Validate the model on valid / test set"""
    
    epoch_log = {'loss': [], 'recons_loss': [], 'kl_div': []}
    for images, bbox, scale, imgfile, label in tqdm(eval_loader, desc='Valid', leave=False):
        model.eval()
        z, recon_images = model(images.to(device))
        loss = model.module.loss_fn(recon_images, images.to(device))
        
        if n_gpu > 1:
            loss = loss.mean()
        epoch_log['loss'].append(loss.item())
        torch.cuda.empty_cache()
       
    result = {}
    for k, v in epoch_log.items():
        result[k] = np.mean(v)
    
    return result

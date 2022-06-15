import os
from tqdm.auto import tqdm
import numpy as np
import torch

from umap import UMAP
import matplotlib.pyplot as plt


def compare_true_and_recons_image(true_image, recons_image):
    plt.figure(figsize=(10, 6))
    plt.subplot(121)
    plt.imshow(true_image.cpu().numpy().transpose((1, 2, 0)))
    plt.title(f'Original - Epoch {epoch}')
    plt.subplot(122)
    plt.imshow(recons_image.detach().cpu().numpy().transpose((1, 2, 0)))
    plt.title(f'Reconstructed - Epoch {epoch}')
    plt.savefig(f'./recons_image/{epoch}.png')

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_latent_vector(model, data_loader, return_mean=True):
    model.eval()

    vector = []
    labels = []
    for images, label in tqdm(data_loader):
        mu = model.get_latent_vector(images.to(device), return_mean)
        vector.append(mu.cpu().detach().numpy())
        labels.append(label.cpu().detach().numpy())
    
    vector = np.vstack(vector)
    labels = np.hstack(labels)
    
    return vector, labels

def vis_vector_umap(vector, labels):
    label_set = np.unique(labels)
    
    colors = ['#aaffaa', '#ffaaff', '#d8d8d8', '#ffff00', '#ecb8db', '#0c1b6f', '#50b883', '#ff6600', '#a26735']
    umap_params = {
        'a1': {'n_neighbors':10, 'min_dist':0.25}, 
        'a2': {'n_neighbors':10, 'min_dist':0.8}, 
        'a3': {'n_neighbors':5, 'min_dist':0.5}, 
        'a4': {'n_neighbors':10, 'min_dist':1}
    }
    
    for k, v in umap_params.items():
        umap = UMAP(**v)
        vector_dr = umap.fit_transform(vector)
        # df = pd.DataFrame({'x': vector_dr[:, 0], 'y': vector_dr[:, 1], 'label': labels})

        fig, ax = plt.subplots(figsize=(8,8))
        for l in label_set:
            mask = (labels == l)
            ax.scatter(x=vector_dr[:, 0][mask], y=vector_dr[:, 1][mask], color=colors[list(label_set).index(l)], label=f'Type {l}')
        plt.legend()


def log_metric(log, epoch, subset):
    out = []
    for metric, value in log.items():
        out.append(f'{metric}: {np.mean(value):.6f}')
    out = f'{subset} Epoch {epoch}: ' + ', '.join(out)
    tqdm.write(out)

def save_ckpt(model, optimizer, epoch, train_log, n_gpu, args):
    model_state_dict = model.state_dict() if n_gpu == 1 else model.module.state_dict()
    state_dict = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_log['loss'][-1],
        'recons_loss': train_log['recons_loss'][-1],
        'kl_div': train_log['kl_div'][-1],
    }
    torch.save(state_dict, os.path.join(args.checkpoint_dir, f'ae_no_bn_{args.latent_dim}_{epoch}.pth'))

def load_data_parallel_model():
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint['model_state_dict'].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

def get_gradient_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def seed_all(seed):
    # np.random.seed(args.random_seed)
    # torch.manual_seed(args.random_seed)
    # torch.backends.cudnn.enabled = False   
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    pass

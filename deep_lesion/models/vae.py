import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.models.autoencoders import VAE
from pl_bolts.models.autoencoders.components import Interpolate
from pl_bolts.models.self_supervised.ssl_finetuner import SSLFineTuner

# from pl_bolts.models.autoencoders.components import resnet18_decoder, resnet18_encoder
# from .components import resnet18_decoder, resnet18_encoder, resnet50_encoder, resnet50_decoder


class LesionVAE(VAE):
    def __init__(self, input_channel, pretrained_model=None, first_conv=False, **vae_kw):
        super().__init__(**vae_kw)

        # Use 7x7 in first conv layer
        if first_conv:
            self.encoder.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # self.decoder.upscale = Interpolate(scale_factor=2)
            # self.decoder.upscale_factor *= 2
        else:
            self.encoder.conv1 = nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
            # self.decoder.upscale = Interpolate(scale_factor=1)
        self.decoder.conv1 = nn.Conv2d(64, input_channel, kernel_size=3, stride=1, padding=1, bias=False)
        
        # self.encoder.conv1 = nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # if self.enc_type.endswith('18'):
        #     explosion = 1
        # else:
        #     explosion = 4
        # self.decoder.conv1 = nn.Conv2d(64 * explosion, input_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.save_hyperparameters()

    def from_pretrained(self, checkpoint_name):
        if checkpoint_name not in VAE.pretrained_urls:
            raise KeyError(str(checkpoint_name) + ' not present in pretrained weights.')

        model = self.load_from_checkpoint(VAE.pretrained_urls[checkpoint_name], strict=False)

        if input_channel != 3:
            model.encoder.conv1 = nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if self.enc_type.endswith('18'):
                explosion = 1
            else:
                explosion = 4
            model.decoder.conv1 = nn.Conv2d(64 * explosion, input_channel, kernel_size=3, stride=1, padding=1, bias=False)
        
        return model

    def get_latent_vector(self, x, return_mean=True):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)

        if return_mean:
            return mu
        else:
            return z

class LesionVAEForFinetune(SSLFineTuner):
    def __init__(self, num_labels=8, image_dim=None, latent_dim=None, use_filter=None, **finetuner_kw):
        super().__init__(**finetuner_kw)
        dropout = finetuner_kw['dropout']

        # rep_dim = finetuner_kw['hidden_dim']
        rep_dim = finetuner_kw['in_features']

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(rep_dim, rep_dim//2, bias=False),
            # nn.BatchNorm1d(rep_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(rep_dim//2, num_labels, bias=True),
        )
    
    def shared_step(self, batch):
        x, y = batch

        # with torch.no_grad():
        feats = self.backbone.encoder(x)
        # feats = self.backbone.get_latent_vector(x)

        feats = feats.view(feats.size(0), -1)
        logits = self.classifier(feats)
        loss = F.cross_entropy(logits, y)

        return loss, logits, y

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            list(self.backbone.parameters()) + list(self.classifier.parameters()),
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

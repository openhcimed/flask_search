import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pl_bolts.models.self_supervised import SimCLR


class SimCLRRMAC(pl.LightningModule):
    def __init__(self, backbone, image_dim=None, latent_dim=None, use_filter=None, dropout=None, margin=None, 
                 mode=None, epochs=None, lr=None, weight_decay=None, scheduler_type=None):
        super().__init__()

        self.backbone = backbone
        self.image_dim = image_dim
        self.latent_dim = latent_dim
        self.user_filter = use_filter
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type

        self.rmac = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, latent_dim),
            # nn.AdaptiveMaxPool2d(output_size=1)
            # GeneralizedMeanPoolingP(norm=gemp)
        )

        self.margin = margin
        self.loss_fct = nn.TripletMarginLoss(margin)

    def forward(self, x):
        # with torch.no_grad():
        # print('shape', x.shape)
        x = self.backbone(x)
        # print('shape', x.shape)
        x = self.rmac(x)
        x = F.normalize(x, p=2, dim=1)

        return x

    def training_step(self, batch, batch_idx):
        query, pos, neg = batch       # (B, 1, image_dim, image_dim), (B, neg_size, image_dim, image_dim)
        batch_size = query.shape[0]
        neg_size = neg.shape[1]

        query_emb = self.forward(query)
        pos_emb = self.forward(pos)
        neg_emb = self.forward(neg)

        loss = self.loss_fct(query_emb, pos_emb, neg_emb)

        self.log_dict({'loss': loss})
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            list(self.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # set scheduler
        if self.scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.epochs, gamma=self.gamma)
        elif self.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                self.epochs,
                eta_min=self.lr / 10  # total epochs to run
            )

        return [optimizer], [scheduler]

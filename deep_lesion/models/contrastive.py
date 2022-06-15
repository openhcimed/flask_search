import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

from pl_bolts.models.self_supervised import SimCLR
from pl_bolts.models.self_supervised.ssl_finetuner import SSLFineTuner


class LesionSimCLR(SimCLR):
    def __init__(self, input_channel, first_conv=False, use_filter=None, **simclr_kw):
        super().__init__(**simclr_kw)
        
        # Use 7x7 in first conv layer
        if first_conv:
            self.encoder.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.encoder.conv1 = nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.save_hyperparameters()
    
    def get_latent_vector(self, x, **kw):
        h = self.forward(x)
        h = self.projection(h)
        
        return h
    
    def shared_step(self, batch):
        if self.dataset == "stl10":
            unlabeled_batch = batch[0]
            batch = unlabeled_batch

        # final image in tuple is for online eval
        (img1, img2, _), y = batch
        print('shape', img1.shape)
        raise

        # get h representations, bolts resnet returns a list
        h1 = self(img1)
        h2 = self(img2)

        # get z representations
        z1 = self.projection(h1)
        z2 = self.projection(h2)

        loss = self.nt_xent_loss(z1, z2, self.temperature)

        return loss

class LesionSimCLRForFinetune(SSLFineTuner):
    def __init__(self, num_labels=None, image_dim=None, latent_dim=None, use_filter=None, **finetuner_kw):
        super().__init__(**finetuner_kw)
        dropout = finetuner_kw['dropout']
        
        # rep_dim = finetuner_kw['hidden_dim']
        rep_dim = finetuner_kw['in_features']

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(rep_dim, rep_dim, bias=False),
            # nn.BatchNorm1d(rep_dim//2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(rep_dim, num_labels, bias=True),
        )

    def on_train_epoch_start(self) -> None:
        self.backbone.train()

    def shared_step(self, batch):
        x, y = batch

        # with torch.no_grad(): # This is the difference from the original implementation
        feats = self.backbone(x)
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

def precision_at_k(label, pred, k=10):
    label_set = set(label)
    pred_set = set(pred[:k])

    return len(label_set & pred_set) / k

def recall_at_k(label, pred, k=10):
    label_set = set(label)
    pred_set = set(pred[:k])
    
    return len(label_set & pred_set) / len(label_set)

class LesionSimCLRForRetrival(SSLFineTuner):
    def __init__(self, image_dim=None, latent_dim=None, use_filter=None, neg_size=None, **finetuner_kw):
        super().__init__(**finetuner_kw)
        dropout = finetuner_kw['dropout']

        # rep_dim = finetuner_kw['hidden_dim']
        rep_dim = finetuner_kw['in_features']
        self.rep_dim = rep_dim

        self.proj = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(rep_dim, rep_dim, bias=False),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=dropout),
            # nn.Linear(rep_dim, rep_dim, bias=True),
        )

        self.neg_size = neg_size
        self.weight = nn.Parameter(torch.ones((1, rep_dim)))
        self.gamma = 1.0

    def on_train_epoch_start(self) -> None:
        self.backbone.train()

    def forward(self, batch):
        pos_head, pos_tail, neg_tails = batch       # (B, 1, image_dim, image_dim), (B, neg_size, image_dim, image_dim)
        batch_size = pos_head.shape[0]

        if self.neg_size > 1:
            # Shape: B, 1, rep_dim
            with torch.no_grad():
                feats_pos_head = self.backbone(pos_head).unsqueeze(dim=1)
                feats_pos_tail = self.backbone(pos_tail).unsqueeze(dim=1)
                feats_neg_tail = self.backbone(neg_tails.view(-1, 1, 64, 64)).view(batch_size, self.neg_size, -1)
            
            # print('shape', feats_neg_tail.shape, self.rep_dim)

            feats_pos_head = self.proj(feats_pos_head)
            feats_pos_tail = self.proj(feats_pos_tail)
            feats_neg_tail = self.proj(feats_neg_tail)

            pos_score = feats_pos_head * self.weight * feats_pos_tail
            neg_score = feats_pos_head * self.weight * feats_neg_tail
            pos_score = torch.norm(pos_score, dim=2)
            neg_score = torch.norm(neg_score, dim=2)

            logits = torch.cat([pos_score, neg_score], dim=1)
            pos_loss = - F.logsigmoid(self.gamma - pos_score).mean()
            neg_loss = - F.logsigmoid(neg_score - self.gamma).mean()

            loss = (pos_loss + neg_loss) / 2
            
            labels = None

        else:
            # Shape: B, 1, rep_dim
            # with torch.no_grad():
            feats_pos_head = self.backbone(pos_head)
            feats_pos_tail = self.backbone(pos_tail)
            feats_neg_tail = self.backbone(neg_tails)
            
            feats_pos_head = self.proj(feats_pos_head)
            feats_pos_tail = self.proj(feats_pos_tail)
            feats_neg_tail = self.proj(feats_neg_tail)

            pos_score = feats_pos_head * self.weight * feats_pos_tail
            neg_score = feats_pos_head * self.weight * feats_neg_tail
            # pos_score = torch.norm(pos_score, dim=2)
            # neg_score = torch.norm(neg_score, dim=2)

            logits = torch.cat([pos_score, neg_score], dim=0).sum(dim=1)
            labels = torch.cat([torch.ones((pos_score.shape[0],)), torch.zeros((neg_score.shape[0],))], dim=0).to(logits.device)

            loss = F.binary_cross_entropy_with_logits(logits, labels)


        return loss, logits, labels
    
    def training_step(self, batch, batch_idx):
        loss, logits, labels = self.forward(batch)

        if batch_idx % 100 == 0:
            print('logits', logits)

        # tensorboard_logs = {"train_loss": loss}

        return {"loss": loss}#, "log": tensorboard_logs}
    
    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self.forward(batch)

        if self.neg_size > 1:
            labels = torch.vstack([torch.arange(logits.shape[1]) for i in range(logits.shape[0])])
            rank = torch.argsort(logits, dim=1)
            mask = (rank == 0)      # First one is positive sample
            ranking = (labels[mask]).float() + 1

            # Calculate Precision@k, Recall@k
            mr = ranking.mean()
            mrr = (1 / ranking).mean()
            hits_at_1 = (ranking <= 1).float().mean()
            hits_at_3 = (ranking <= 3).float().mean()
            hits_at_5 = (ranking <= 5).float().mean()
            hits_at_10 = (ranking <= 10).float().mean()

            self.log_dict({
                'MR': mr,
                'MRR': mrr,
                'val_hits@1': hits_at_1, 
                'val_hits@3': hits_at_3, 
                'val_hits@5': hits_at_5, 
                'val_hits@10': hits_at_10, 
            })

        else:
            labels = labels.cpu().numpy()
            logits = logits.detach().cpu().numpy()#.mean(axis=1)
            auroc = roc_auc_score(labels, logits)
            auc_pr = average_precision_score(labels, logits)

            if batch_idx % 100 == 0:
                print('logits', logits)
            
            pred = np.argsort(logits)
            label_set = np.arange(logits.shape[0]//2, dtype=np.int)
            
            precision_at_1 = precision_at_k(label_set, pred, k=1)
            precision_at_3 = precision_at_k(label_set, pred, k=3)
            precision_at_5 = precision_at_k(label_set, pred, k=5)
            precision_at_10 = precision_at_k(label_set, pred, k=10)
            recall_at_1 = recall_at_k(label_set, pred, k=1)
            recall_at_3 = recall_at_k(label_set, pred, k=3)
            recall_at_5 = recall_at_k(label_set, pred, k=5)
            recall_at_10 = recall_at_k(label_set, pred, k=10)

            self.log_dict({
                'AUC-PR': auc_pr, 
                'Precision@1': precision_at_1,
                'Precision@3': precision_at_3,
                'Precision@5': precision_at_5,
                'Precision@10': precision_at_10,
                'Recall@1': recall_at_1,
                'Recall@3': recall_at_3,
                'Recall@5': recall_at_5,
                'Recall@10': recall_at_10,
            })
    
    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            list(self.parameters()),
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

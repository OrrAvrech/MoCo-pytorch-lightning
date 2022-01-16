import torch
from tqdm import tqdm
from torchmetrics import Accuracy
from params import Params
from torchvision.models import resnet50
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F


class Backbone(nn.Module):
    def __init__(self, feature_dim=128, add_mlp_head=False):
        super(Backbone, self).__init__()

        backbone = resnet50(num_classes=feature_dim)
        if add_mlp_head:
            num_filters = backbone.fc.in_features
            layers = list(backbone.children())[:-1]
            mlp = nn.Sequential(nn.Linear(num_filters, 2048),
                                nn.ReLU(inplace=True),
                                nn.Linear(2048, feature_dim))
            net = nn.Sequential(*layers, *mlp.children())

        else:
            net = backbone
        self.net = net

    def forward(self, x):
        x = self.net(x)
        return x


class LitMoCo(pl.LightningModule):
    def __init__(self, dim=128, k=4096, m=0.99, t=0.07, add_mlp_head=False, bank_data_loader=None):
        super(LitMoCo, self).__init__()
        self.save_hyperparameters()

        self.k = k
        self.m = m
        self.t = t
        # bank_data_loader for KNN evaluation
        self.bank_data_loader = bank_data_loader
        self.feature_bank = torch.zeros((dim, Params.MoCo.BATCH_SIZE))
        self.feature_labels = torch.zeros((Params.MoCo.BATCH_SIZE,))
        self.accuracy = Accuracy()

        # create the encoders
        self.encoder_q = Backbone(feature_dim=dim, add_mlp_head=add_mlp_head)
        self.encoder_k = Backbone(feature_dim=dim, add_mlp_head=add_mlp_head)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, self.k))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def _update_queue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.k % batch_size == 0

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()  # transpose
        ptr = (ptr + batch_size) % self.k  # move pointer

        self.queue_ptr[0] = ptr

    def _contrastive_loss(self, im_q, im_k):
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():
            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

        # positive logits: Nx1
        # scalar product for each element in batch
        l_pos = torch.sum((q * k), dim=1).unsqueeze(-1)
        # negative logits: NxK
        # matrix mul between query features and queue keys
        l_neg = torch.matmul(q, self.queue.clone().detach())

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1) / self.t

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = nn.CrossEntropyLoss().cuda()(logits, labels)

        return loss, q, k

    def forward(self, im1, im2):
        # update the momentum key encoder without backprop
        with torch.no_grad():
            self._momentum_update_key_encoder()

        # contrastive loss
        loss, q, k = self._contrastive_loss(im1, im2)

        with torch.no_grad():
            self._update_queue(k)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(),
                                    lr=Params.MoCo.LR,
                                    weight_decay=Params.MoCo.WEIGHT_DECAY,
                                    momentum=Params.MoCo.MOMENTUM)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, Params.MoCo.EPOCHS)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        im1, im2 = batch
        loss = self(im1, im2)
        self.log('train_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        if self.bank_data_loader is not None:
            feature_bank, memory_targets = [], []
            with torch.no_grad():
                # generate feature bank
                for data, target in tqdm(self.bank_data_loader, desc='Feature extracting'):
                    feature = self.encoder_q(data.cuda(non_blocking=True))
                    feature = F.normalize(feature, dim=1)
                    feature_bank.append(feature)
                    memory_targets.append(target)
                # [D, N]
                feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
                # [N]
                feature_labels = torch.cat(memory_targets, dim=0).clone().detach().to(feature_bank.device)
            self.feature_bank = feature_bank
            self.feature_labels = feature_labels

    def _knn_predict(self, feature, classes, knn_k, knn_t):
        # compute cos similarity between each feature vector and feature bank ---> [B, N]
        sim_matrix = torch.mm(feature, self.feature_bank)
        # [B, K]
        sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
        # [B, K]
        sim_labels = torch.gather(self.feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
        sim_weight = (sim_weight / knn_t).exp()

        # counts for each class
        one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
        # [B*K, C]
        one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
        # weighted score ---> [B, C]
        pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

        pred_labels = pred_scores.argsort(dim=-1, descending=True)
        return pred_labels

    def validation_step(self, batch, batch_idx):
        if self.bank_data_loader is not None:
            classes = len(self.bank_data_loader.dataset.classes)
            data, target = batch
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = self.encoder_q(data)
            feature = F.normalize(feature, dim=1)
            pred_labels = self._knn_predict(feature, classes, Params.KNN.K, Params.KNN.T)
            self.accuracy(pred_labels[:, 0], target)
            self.log('top1_acc', self.accuracy, on_step=False, on_epoch=True, prog_bar=True)


# class LitMoCo(pl.LightningModule):
#     def __init__(self, dim=128, k=4096, m=0.99, t=0.07):
#         super(LitMoCo, self).__init__()
#         self.save_hyperparameters()
#
#         self.k = k
#         self.m = m
#         self.t = t
#
#         # encoders
#         self.encoder_q = resnet50(num_classes=dim)
#         self.encoder_k = resnet50(num_classes=dim)
#
#         for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
#             param_k.data.copy_(param_q.data)
#             param_k.requires_grad = False
#
#         # queue
#         self.register_buffer("queue", torch.randn(dim, self.k))
#         self.queue = nn.functional.normalize(self.queue, dim=0)
#
#         self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
#
#     def _momentum_update_key_encoder(self):
#         for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
#             param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
#
#     def _update_queue(self, keys):
#         batch_size = keys.shape[0]
#
#         ptr = int(self.queue_ptr)
#         assert self.k % batch_size == 0
#
#         # replace the keys at ptr (dequeue and enqueue)
#         self.queue[:, ptr:ptr + batch_size] = keys.t()  # transpose
#         ptr = (ptr + batch_size) % self.k  # move pointer
#
#         self.queue_ptr[0] = ptr
#
#     def _contrastive_loss(self, im_q, im_k):
#         # compute query features
#         q = self.encoder_q(im_q)  # queries: NxC
#         q = nn.functional.normalize(q, dim=1)
#
#         # compute key features
#         with torch.no_grad():
#             k = self.encoder_k(im_k)  # keys: NxC
#             k = nn.functional.normalize(k, dim=1)
#
#         # positive logits: Nx1
#         # scalar product for each element in batch
#         l_pos = torch.sum((q * k), dim=1).unsqueeze(-1)
#         # negative logits: NxK
#         # matrix mul between query features and queue keys
#         l_neg = torch.matmul(q, self.queue.clone().detach())
#
#         # logits: Nx(1+K)
#         logits = torch.cat([l_pos, l_neg], dim=1) / self.t
#
#         # labels: positive key indicators
#         labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
#         loss = nn.CrossEntropyLoss().cuda()(logits, labels)
#
#         return loss, q, k
#
#     def forward(self, im1, im2):
#         # update the momentum key encoder without backprop
#         with torch.no_grad():
#             self._momentum_update_key_encoder()
#
#         # contrastive loss
#         loss, q, k = self._contrastive_loss(im1, im2)
#
#         with torch.no_grad():
#             self._update_queue(k)
#
#         return loss
#
#     def configure_optimizers(self):
#         optimizer = torch.optim.SGD(self.parameters(),
#                                     lr=Params.MoCo.LR,
#                                     weight_decay=Params.MoCo.WEIGHT_DECAY,
#                                     momentum=Params.MoCo.MOMENTUM)
#         lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, Params.MoCo.EPOCHS)
#         return [optimizer], [lr_scheduler]
#
#     def training_step(self, batch, batch_idx):
#         im1, im2 = batch
#         loss = self(im1, im2)
#         self.log('train_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
#         return loss


class LitLinearClassifier(pl.LightningModule):
    def __init__(self, num_classes, pre_trained=True, ckpt_path=None):
        super(LitLinearClassifier, self).__init__()
        if ckpt_path is not None:
            moco = LitMoCo.load_from_checkpoint(checkpoint_path=ckpt_path)
            backbone = moco.encoder_q.net
        else:
            backbone = resnet50(pretrained=pre_trained)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(num_filters, num_classes)
        self.accuracy = Accuracy()

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(),
                                    lr=Params.Classifier.LR,
                                    weight_decay=Params.Classifier.WEIGHT_DECAY,
                                    momentum=Params.Classifier.MOMENTUM)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, Params.Classifier.EPOCHS)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        pred = torch.argmax(F.log_softmax(y_hat, dim=1), dim=1)
        self.accuracy(pred, y)
        self.log('val_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log('val_acc', self.accuracy, on_epoch=True, on_step=False, prog_bar=True)
        return loss

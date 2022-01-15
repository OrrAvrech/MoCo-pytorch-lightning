import torch
from tqdm import tqdm
from torchmetrics import Accuracy
from params import Params
from torchvision.models import resnet, resnet50
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F


class BackboneModel(nn.Module):
    def __init__(self, feature_dim=128, arch=None):
        super(BackboneModel, self).__init__()

        norm_layer = nn.BatchNorm2d
        resnet_arch = getattr(resnet, arch)
        net = resnet_arch(num_classes=feature_dim, norm_layer=norm_layer)
        self.net = net

    def forward(self, x):
        x = self.net(x)
        return x


class MoCo(nn.Module):
    def __init__(self, dim=128, k=4096, m=0.99, t=0.07, symmetric=True):
        super(MoCo, self).__init__()

        self.k = k
        self.m = m
        self.t = t
        self.symmetric = symmetric

        # create the encoders
        # self.encoder_q = BackboneModel(feature_dim=dim, arch=arch)
        # self.encoder_k = BackboneModel(feature_dim=dim, arch=arch)
        self.encoder_q = resnet50(num_classes=dim)
        self.encoder_k = resnet50(num_classes=dim)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, self.k))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.k % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()  # transpose
        ptr = (ptr + batch_size) % self.k  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]

    def contrastive_loss(self, im_q, im_k):
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)  # already normalized

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)

            k = self.encoder_k(im_k_)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)  # already normalized

            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.t

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        loss = nn.CrossEntropyLoss().cuda()(logits, labels)

        return loss, q, k

    def forward(self, im1, im2):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """

        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()

        # compute loss
        if self.symmetric:  # asymmetric loss
            loss_12, q1, k2 = self.contrastive_loss(im1, im2)
            loss_21, q2, k1 = self.contrastive_loss(im2, im1)
            loss = loss_12 + loss_21
            k = torch.cat([k1, k2], dim=0)
        else:  # asymmetric loss
            loss, q, k = self.contrastive_loss(im1, im2)

        self._dequeue_and_enqueue(k)

        return loss


class LitMoCo(pl.LightningModule):
    def __init__(self, dim=128, k=4096, m=0.99, t=0.07, symmetric=True, bank_data_loader=None):
        super(LitMoCo, self).__init__()

        self.k = k
        self.m = m
        self.t = t
        self.symmetric = symmetric
        # bank_data_loader for KNN evaluation
        self.bank_data_loader = bank_data_loader
        self.feature_bank = torch.zeros((dim, self.bank_data_loader.batch_size))
        self.feature_labels = torch.zeros((self.bank_data_loader.batch_size,))
        self.accuracy = Accuracy()

        # create the encoders
        self.encoder_q = resnet50(num_classes=dim)
        self.encoder_k = resnet50(num_classes=dim)

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

    def _dequeue_and_enqueue(self, keys):
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
        q = nn.functional.normalize(q, dim=1)  # already normalized

        # compute key features
        with torch.no_grad():
            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)  # already normalized

        # compute logits
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.t

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = nn.CrossEntropyLoss().cuda()(logits, labels)

        return loss, q, k

    def forward(self, im1, im2):
        # update the momentum key encoder without backprop
        with torch.no_grad():
            self._momentum_update_key_encoder()

        # compute loss
        if self.symmetric:  # asymmetric loss
            loss_12, q1, k2 = self._contrastive_loss(im1, im2)
            loss_21, q2, k1 = self._contrastive_loss(im2, im1)
            loss = loss_12 + loss_21
            k = torch.cat([k1, k2], dim=0)
        else:  # asymmetric loss
            loss, q, k = self._contrastive_loss(im1, im2)

        with torch.no_grad():
            self._dequeue_and_enqueue(k)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(),
                                    lr=Params.MoCo.LR,
                                    weight_decay=Params.MoCo.WEIGHT_DECAY,
                                    momentum=Params.MoCo.MOMENTUM)
        return optimizer

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


class LitLinearClassifier(pl.LightningModule):
    def __init__(self, num_classes, pre_trained=True, ckpt_path=None):
        super(LitLinearClassifier, self).__init__()
        if ckpt_path is not None:
            moco = MoCo().load_state_dict(torch.load(ckpt_path))
            backbone = moco.encoder_q
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
        return optimizer

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

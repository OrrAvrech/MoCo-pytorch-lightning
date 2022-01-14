import torch
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from models import MoCo
from params import Params
from datasets import ImagenettePair
import pandas as pd
from torch.nn import functional as F


def train(net, data_loader, optimizer, epoch):
    net.train()

    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for im_1, im_2 in train_bar:
        im_1, im_2 = im_1.cuda(non_blocking=True), im_2.cuda(non_blocking=True)

        loss = net(im_1, im_2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size
        train_bar.set_description(
            'Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}'.format(epoch, Params.MoCo.EPOCHS,
                                                                    optimizer.param_groups[0]['lr'],
                                                                    total_loss / total_num))

    return total_loss / total_num


def test(net, memory_data_loader, test_data_loader, epoch):
    net.eval()
    classes = len(memory_data_loader.dataset.classes)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature = net(data.cuda(non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)

            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, Params.KNN.K, Params.KNN.T)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            test_bar.set_description(
                'Test Epoch: [{}/{}] Acc@1:{:.2f}%'.format(epoch, Params.MoCo.EPOCHS, total_top1 / total_num * 100))

    return total_top1 / total_num * 100


def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


def main():

    # data loaders
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    train_ds = ImagenettePair(transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=Params.MoCo.BATCH_SIZE, shuffle=True, drop_last=True)

    memory_ds = ImagenettePair(transform=test_transform)
    memory_loader = DataLoader(memory_ds, batch_size=Params.MoCo.BATCH_SIZE, shuffle=False)

    test_ds = ImagenettePair(split='val', transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=Params.MoCo.BATCH_SIZE, shuffle=False)

    # create model
    model = MoCo(
        dim=Params.MoCo.DIM,
        k=Params.MoCo.K,
        m=Params.MoCo.M,
        t=Params.MoCo.T,
        arch=Params.MoCo.ARCH,
        symmetric=Params.MoCo.SYMMETRIC,
    ).cuda()

    # logging
    results = {'train_loss': [], 'test_acc@1': []}
    Path(Params.RESULTS_DIR).mkdir(exist_ok=True)

    # training loop
    epoch_start = 1
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=Params.MoCo.LR,
                                weight_decay=Params.MoCo.WEIGHT_DECAY,
                                momentum=Params.MoCo.MOMENTUM)
    for epoch in range(epoch_start, Params.MoCo.EPOCHS + 1):
        train_loss = train(model, train_loader, optimizer, epoch)
        results['train_loss'].append(train_loss)
        test_acc_1 = test(model.encoder_q, memory_loader, test_loader, epoch)
        results['test_acc@1'].append(test_acc_1)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(epoch_start, epoch + 1))
        data_frame.to_csv(Params.RESULTS_DIR + '/log.csv', index_label='epoch')
        # save model
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), },
                   Params.RESULTS_DIR + '/model_last.pth')


if __name__ == '__main__':
    main()

from params import Params
from datasets import Imagenette, ImagenettePair
from models import LitMoCo
from torchvision import transforms
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger


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
        transforms.CenterCrop(32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    train_ds = ImagenettePair(transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=Params.MoCo.BATCH_SIZE, shuffle=True, drop_last=True)

    bank_ds = Imagenette(transform=test_transform)
    bank_loader = DataLoader(bank_ds, batch_size=Params.MoCo.BATCH_SIZE, shuffle=False)

    test_ds = Imagenette(split='val', transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=Params.MoCo.BATCH_SIZE, shuffle=False)

    moco = LitMoCo(dim=Params.MoCo.DIM,
                   k=Params.MoCo.K,
                   m=Params.MoCo.M,
                   t=Params.MoCo.T,
                   symmetric=Params.MoCo.SYMMETRIC,
                   bank_data_loader=bank_loader).cuda()

    # callbacks
    csv_logger = CSVLogger(save_dir=Params.RESULTS_DIR, name='pl_logs_moco')

    trainer = Trainer(logger=csv_logger, gpus=1, max_epochs=Params.MoCo.EPOCHS)
    trainer.fit(moco, train_loader, test_loader)


if __name__ == '__main__':
    main()

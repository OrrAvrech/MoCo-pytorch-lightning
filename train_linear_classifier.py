from params import Params
from datasets import Imagenette
from models import LitLinearClassifier
from torchvision import transforms
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger


def main():
    # data loaders
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(128),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    num_classes = Imagenette.get_num_classes()
    train_ds = Imagenette(transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=Params.Classifier.BATCH_SIZE, shuffle=True)

    val_ds = Imagenette(split='val', transform=test_transform)
    val_loader = DataLoader(val_ds, batch_size=Params.Classifier.BATCH_SIZE, shuffle=False)

    classifier = LitLinearClassifier(num_classes=num_classes)

    # callbacks
    csv_logger = CSVLogger(save_dir=Params.RESULTS_DIR, name='pl_logs')

    trainer = Trainer(logger=csv_logger, gpus=1, max_epochs=Params.Classifier.EPOCHS)
    trainer.fit(classifier, train_loader, val_loader)


if __name__ == '__main__':
    main()

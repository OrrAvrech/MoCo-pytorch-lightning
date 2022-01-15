from params import Params
from datasets import Imagenette
from models import LitLinearClassifier
from torchvision import transforms
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def main():
    # data loaders
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(Params.INPUT_SIZE),
        transforms.ToTensor()])

    test_transform = transforms.Compose([
        transforms.Resize(Params.INPUT_SIZE + 10),
        transforms.CenterCrop(Params.INPUT_SIZE),
        transforms.ToTensor()])

    num_classes = Imagenette.get_num_classes()
    train_ds = Imagenette(transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=Params.Classifier.BATCH_SIZE, shuffle=True)

    val_ds = Imagenette(split='val', transform=test_transform)
    val_loader = DataLoader(val_ds, batch_size=Params.Classifier.BATCH_SIZE, shuffle=False)

    classifier = LitLinearClassifier(num_classes=num_classes, ckpt_path=Params.SSL_CKPT_PATH)

    # callbacks
    csv_logger = CSVLogger(save_dir=Params.RESULTS_DIR, name='pl_logs_classifier_moco')
    early_stop_cb = EarlyStopping(monitor='val_loss', patience=6)

    trainer = Trainer(logger=csv_logger, gpus=1, max_epochs=Params.Classifier.EPOCHS, callbacks=[early_stop_cb])
    trainer.fit(classifier, train_loader, val_loader)


if __name__ == '__main__':
    main()

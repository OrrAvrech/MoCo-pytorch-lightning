from PIL import Image
from torch.utils.data import Dataset
from fastai.vision.all import untar_data, URLs, get_image_files, parent_label


class Imagenette(Dataset):

    lbl_dict = dict(
        n01440764='tench',
        n02102040='English springer',
        n02979186='cassette player',
        n03000684='chain saw',
        n03028079='church',
        n03394916='French horn',
        n03417042='garbage truck',
        n03425413='gas pump',
        n03445777='golf ball',
        n03888257='parachute'
    )

    def __init__(self, split='train', transform=None):
        path = untar_data(URLs.IMAGENETTE_160)
        self.file_names = get_image_files(path, folders=split)
        self.transform = transform
        self.split = split
        self.label_dict = self.get_label_dict()
        self.classes = list(self.label_dict.values())
        self.targets = [i for i, x in enumerate(self.classes)]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        f_name = self.file_names[idx]
        image = Image.open(f_name)
        label = self.classes.index(self.label_dict[parent_label(f_name)])
        return image, label

    @classmethod
    def get_label_dict(cls):
        return cls.lbl_dict


class ImagenettePair(Imagenette):
    def __getitem__(self, idx):
        f_name = self.file_names[idx]
        image = Image.open(f_name).convert('RGB')
        image_1 = image
        image_2 = image
        if self.transform is not None:
            image_1 = self.transform(image)
            image_2 = self.transform(image)

        return image_1, image_2

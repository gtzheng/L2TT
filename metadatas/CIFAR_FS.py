from .utils import *
import torchvision.transforms as T
import torch
import numpy as np
filepath,_ = os.path.split(os.path.realpath(__file__))
filepath,_ = os.path.split(filepath)
filepath,_ = os.path.split(filepath)


# Set the appropriate paths of the datasets here.
from config import _CIFAR_FS_DATASET_DIR



class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

class CIFAR_FS(ProtoData):
    def __init__(self, phase='train', augment='null',mean=[x / 255.0 for x in [129.37731888, 124.10583864, 112.47758569]], std=[x / 255.0 for x in [68.20947949, 65.43124043, 70.45866994]]):

        # assert (phase == 'train' or phase == 'final' or phase == 'val' or phase == 'test')
        self.phase = phase
        self.name = 'CIFAR_FS_' + phase
        self.img_size = (32, 32)

        print('Loading CIFAR-FS dataset - phase {0}'.format(phase))
        file_train_categories_train_phase = os.path.join(
            _CIFAR_FS_DATASET_DIR,
            'CIFAR_FS_train.pickle')
        file_val_categories_val_phase = os.path.join(
            _CIFAR_FS_DATASET_DIR,
            'CIFAR_FS_val.pickle')
        file_test_categories_test_phase = os.path.join(
            _CIFAR_FS_DATASET_DIR,
            'CIFAR_FS_test.pickle')
        if self.phase == 'train' or self.phase == 'train_train' or self.phase == 'train_val':
            # During training phase we only load the training phase images
            # of the training categories (aka base categories).
            data = load_data(file_train_categories_train_phase)
        elif 'val' in self.phase:
            data = load_data(file_val_categories_val_phase)
        elif self.phase == 'test':
            data = load_data(file_test_categories_test_phase)
        else:
            raise ValueError('Not valid phase {0}'.format(self.phase))

        self.data = data['data']
        self.labels = data['labels']
        if self.phase == 'train_train':
            num = len(self.data)
            np.random.seed(0)
            indexes = np.arange(num)
            np.random.shuffle(indexes)
            indexes = indexes[0:int(num*0.8)]
            self.data = self.data[indexes]
            self.labels = np.array(self.labels)[indexes]
        elif self.phase == 'train_val':
            num = len(self.data)
            np.random.seed(0)
            indexes = np.arange(num)
            np.random.shuffle(indexes)
            indexes = indexes[int(num*0.8):]
            self.data = self.data[indexes]
            self.labels = np.array(self.labels)[indexes]


        self.label2ind = buildLabelIndex(self.labels)
        self.labelIds = sorted(self.label2ind.keys())

        self.num_cats = len(self.labelIds)

       
        self.mean = mean
        self.std = std
        normalize = transforms.Normalize(mean=mean, std=std)
        self.normalize = normalize
        self.augment = 'null' if self.phase == 'test' or self.phase == 'val' else augment
        if self.augment == 'null':
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])
        elif self.augment == 'norm' or self.augment == 'norm_cutmix':
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        elif self.augment == 'auto_augment':
            policy = T.AutoAugmentPolicy.CIFAR10
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                T.AutoAugment(policy),
                transforms.ToTensor(),
                Cutout(n_holes=1, length=16),
                normalize
            ])
        elif self.augment == 'auto_augment_cutmix':
            policy = T.AutoAugmentPolicy.CIFAR10
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                T.AutoAugment(policy),
                transforms.ToTensor(),
                normalize
            ])
            

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

        

    def __len__(self):
        return len(self.data)

    @property
    def dataset_dir(self):
        return _CIFAR_FS_DATASET_DIR

    def __repr__(self):
        string = self.__class__.__name__ + '(' \
               + 'phase=' + str(self.phase) + ', ' \
               + 'augment=' + str(self.augment)
        if self.augment == 'w_rot90':
            string += ', ' + str(self.rot90)
        string += ')'
        return string

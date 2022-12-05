import os
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

class DogDataset:
    """
    Dog Dataset.
    """
    def __init__(self, batch_size=4, dataset_path='/content/data/images/dogs', if_resize=True):
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.if_resize = if_resize
        self.train_dataset = self.get_train_numpy()
        self.x_mean, self.x_std = self.compute_train_statistics()
        self.transform = self.get_transforms()
        self.train_loader, self.val_loader = self.get_dataloaders()

    def get_train_numpy(self):
        train_dataset = torchvision.datasets.ImageFolder(os.path.join(self.dataset_path, 'train'))
        train_x = np.zeros((len(train_dataset), 224, 224, 3))
        for i, (img, _) in enumerate(train_dataset):
            train_x[i] = img
        return train_x / 255.0

    def compute_train_statistics(self):
        x_mean = None  # per-channel mean
        x_std = None  # per-channel std
        x_mean = np.ones(3)
        x_std = np.zeros(3)
        x_mean[0] = np.mean(self.train_dataset[:, :, :, 0])
        x_mean[1] = np.mean(self.train_dataset[:, :, :, 1])
        x_mean[2] = np.mean(self.train_dataset[:, :, :, 2])


        x_std[0] = np.std(self.train_dataset[:, :, :, 0])
        x_std[1] = np.std(self.train_dataset[:, :, :, 1])
        x_std[2] = np.std(self.train_dataset[:, :, :, 2])
        return x_mean, x_std

    def get_transforms(self):
        if self.if_resize:
                img_resize = transforms.Resize((32,32))
                imgtopytorchtensor = transforms.ToTensor()
                normalize_val = transforms.Normalize(self.x_mean,self.x_std)
                transform_list = [img_resize,imgtopytorchtensor,normalize_val]
        else:
                imgtopytorchtensor = transforms.ToTensor()
                normalize_val = transforms.Normalize(self.x_mean,self.x_std)
                transform_list = [imgtopytorchtensor,normalize_val]

        transform = transforms.Compose(transform_list)
        return transform

    def get_dataloaders(self):
        train_set = torchvision.datasets.ImageFolder(os.path.join(self.dataset_path, 'train'), transform=self.transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True)

        val_set = torchvision.datasets.ImageFolder(os.path.join(self.dataset_path, 'val'), transform=self.transform)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader

    def plot_image(self, image, label):
        image = np.transpose(image.numpy(), (1, 2, 0))
        image = image * self.x_std.reshape(1, 1, 3) + self.x_mean.reshape(1, 1, 3)  # un-normalize
        plt.title(label)
        plt.imshow((image*255).astype('uint8'))
        plt.show()

    def get_semantic_label(self, label):
        mapping = {'African': 0,
            'Chihuahua': 1,
            'Dhole': 2,
            'Dingo': 3,
            'Japanese Spaniel': 4}
        reverse_mapping = {v: k for k, v in mapping.items()}
        return reverse_mapping[label]


class DogCatDataset:
    """
    Cat vs. Dog Dataset.
    """
    def __init__(self, batch_size=4, dataset_path='/content/data/images/dogs_vs_cats_imbalance'):
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.transform = self.get_transforms()
        self.train_loader, self.val_loader = self.get_dataloaders()

    def get_transforms(self):

        img_resize2 = transforms.Resize((256,256))
        img_crop = transforms.CenterCrop(224)
        imgtopytorchtensor2 = transforms.ToTensor

        normalize_val2 =transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        transform_list = [img_resize2,img_crop,imgtopytorchtensor2,normalize_val2]
        transform = transforms.Compose(transform_list)
        return transform

    def get_dataloaders(self):
        train_set = torchvision.datasets.ImageFolder(os.path.join(self.dataset_path, 'train'), transform=self.transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True)

        # validation set
        val_set = torchvision.datasets.ImageFolder(os.path.join(self.dataset_path, 'val'), transform=self.transform)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader


if __name__ == '__main__':
  dataset = DogDataset()
  print(dataset.x_mean, dataset.x_std)
  images, labels = iter(dataset.train_loader).next()
  # print(images, labels)
  dataset.plot_image(
    torchvision.utils.make_grid(images),
    ', '.join([dataset.get_semantic_label(label.item()) for label in labels])
  )

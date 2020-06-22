#訓練用データセット
#ここのパスは自分のGoogleDriveのパスに合うように変えてください

from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms

class Mydatasets(Dataset):
    def __init__(self, path1,path2,train = True):
        self.transform1 = self.getTransform()
        self.transform2 = self.getTransform()
        self.train = train
        self.path1 = path1
        self.path2 = path2
        self.file1 = os.listdir(self.path1)
        self.file2 = os.listdir(self.path2)

        self.datanum = len(self.file1)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        image1 = Image.open(self.path1+"/"+self.file1[idx])
        image2 = Image.open(self.path2+"/"+self.file2[idx])
        image1 = image1.convert('RGB')
        image2 = image2.convert('RGB')
        image1 = self.transform1(image1)
        image2 = self.transform2(image2)
        return image1, image2

    def getTransform(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))])

        transform=transforms.Compose([
            transforms.RandomResizedCrop(128, scale=(1.0, 1.0), ratio=(1., 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        return transform

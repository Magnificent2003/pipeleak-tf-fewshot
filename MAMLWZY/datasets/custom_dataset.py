import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class FewShotDataset(Dataset):

    def __init__(self,
                 root,
                 n_way,
                 n_shot,
                 n_query,
                 n_episode):

        self.root = root
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_episode = n_episode

        self.transform = transforms.Compose([
            transforms.Resize((84,84)),
            transforms.ToTensor()
        ])

        self.classes = os.listdir(root)
        self.n_classes = len(self.classes)

        self.data = {}
        for c in self.classes:
            path = os.path.join(root,c)
            imgs = [os.path.join(path,i) for i in os.listdir(path)]
            self.data[c] = imgs

    def __len__(self):
        return self.n_episode

    def __getitem__(self, idx):

        classes = random.sample(self.classes,self.n_way)

        x_shot=[]
        x_query=[]
        y_shot=[]
        y_query=[]

        for i,c in enumerate(classes):

            imgs=random.sample(self.data[c],self.n_shot+self.n_query)

            shot=imgs[:self.n_shot]
            query=imgs[self.n_shot:]

            for s in shot:
                img=self.transform(Image.open(s).convert('RGB'))
                x_shot.append(img)
                y_shot.append(i)

            for q in query:
                img=self.transform(Image.open(q).convert('RGB'))
                x_query.append(img)
                y_query.append(i)

        x_shot=torch.stack(x_shot)
        x_query=torch.stack(x_query)
        y_shot=torch.tensor(y_shot)
        y_query=torch.tensor(y_query)

        return x_shot,x_query,y_shot,y_query
import numpy as np
import pandas as pd
from PIL import Image
import os
import json

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

BATCH = 16
EPOCHS = 2

LR = 0.01
IM_SIZE = 32

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TRAIN_DIR = '../input/herbarium-2021-fgvc8/train/'
TEST_DIR = '../input/herbarium-2021-fgvc8/test/'

%%time

with open(TRAIN_DIR + 'metadata.json', "r", encoding="ISO-8859-1") as file:
    train = json.load(file)


train_img = pd.DataFrame(train['images'])
train_ann = pd.DataFrame(train['annotations']).drop(columns='image_id')
train_df = train_img.merge(train_ann, on='id')

print(len(train_df))

NUM_CL = len(train_df['category_id'].value_counts())
tr_df = train_df[:1000]
X_Train, Y_Train = tr_df['file_name'].values, tr_df['category_id'].values
Transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((IM_SIZE, IM_SIZE)),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


class GetData(Dataset):
    def __init__(self, Dir, FNames, Labels, Transform):
        self.dir = Dir
        self.fnames = FNames
        self.transform = Transform
        self.labels = Labels

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        x = Image.open(os.path.join(self.dir, self.fnames[index]))

        if "train" in self.dir:
            return self.transform(x), self.labels[index]
        elif "test" in self.dir:
            return self.transform(x), self.fnames[index]


trainset = GetData(TRAIN_DIR, X_Train, Y_Train, Transform)
trainloader = DataLoader(trainset, batch_size=BATCH, shuffle=True)
next(iter(trainloader))[0].shape

model = torchvision.models.resnet34()
model.fc = nn.Linear(512, NUM_CL, bias=True)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# pip install torchinfo
# from torchinfo import summary

# summary(model)
%%time

for epoch in range(EPOCHS):
    tr_loss = 0.0

    model = model.train()

    for i, (images, labels) in enumerate(trainloader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        logits = model(images.float())
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tr_loss += loss.detach().item()

    model.eval()
    print('Epoch: %d | Loss: %.4f' % (epoch, tr_loss / i))

%%time

with open(TEST_DIR + 'metadata.json', "r", encoding="ISO-8859-1") as file:
    test = json.load(file)

test_df = pd.DataFrame(test['images'])
print(len(test_df))

X_Test = test_df['file_name'].values
testset = GetData(TEST_DIR, X_Test, None, Transform)
testloader = DataLoader(testset, batch_size=1, shuffle=False)

%%time

s_ls = []

with torch.no_grad():
    model.eval()
    for image, fname in testloader:
        image = image.to(DEVICE)

        logits = model(image)
        ps = torch.exp(logits)
        _, top_class = ps.topk(1, dim=1)

        for pred in top_class:
            s_ls.append([fname[0].split('/')[-1][:-4], pred.item()])


sub = pd.DataFrame.from_records(s_ls, columns=['Id', 'Predicted'])
sub.to_csv("submission.csv", index=False)
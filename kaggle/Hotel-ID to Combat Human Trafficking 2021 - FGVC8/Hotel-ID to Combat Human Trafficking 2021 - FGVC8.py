import numpy as np
import pandas as pd
from PIL import Image
import os

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

path = '../input/hotel-id-2021-fgvc8/'
TRAIN_DIR = path + 'train_images/'
TEST_DIR = path + 'test_images/'

train_df = pd.read_csv('../input/hotel-id-2021-fgvc8/train.csv')
train_df[train_df['chain']==8]

NUM_CL = len(train_df['hotel_id'].value_counts())
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(train_df['hotel_id'])
train_df['label'] = le.transform(train_df['hotel_id'])
class_map = dict(sorted(train_df[['label', 'hotel_id']].values.tolist()))
train_df[train_df['label'] == 6754]
train_df['chain_image'] = train_df['chain'].astype(str) + '/' + train_df['image']

tr_df = train_df[:1000]
print(len(tr_df))
X_Train, Y_Train = tr_df['chain_image'].values, tr_df['label'].values

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

X_Test = [name for name in (os.listdir(TEST_DIR))]
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
            s_ls.append([fname[0], pred.item()])


pred_df = pd.DataFrame.from_records(s_ls, columns=['image', 'label'])
pred_df['hotel_id'] = pred_df['label'].map(class_map)
sub = pred_df[['image', 'hotel_id']]
sub.to_csv("submission.csv", index=False)




import sys

# import timm
sys.path.append('input/pytorch-image-models-master')

import numpy as np
import pandas as pd

import math
import random
import os
import cv2
import timm

from tqdm import tqdm

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
from torch.utils.data import Dataset
from torch import nn
import torch.nn.functional as F

import gc
import cudf
import cuml
import cupy
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.neighbors import NearestNeighbors


class CFG:
    img_size = 512
    batch_size = 12
    seed = 2021

    device = 'cuda'
    classes = 11014

    scale = 30
    margin = 0.5


def read_dataset():
    df = pd.read_csv('input/shopee-product-matching/test.csv')
    df_cu = cudf.DataFrame(df)
    image_paths = 'input/shopee-product-matching/test_images/' + df['image']
    return df, df_cu, image_paths


def seed_torch(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_torch(CFG.seed)


def combine_predictions(row):
    x = np.concatenate([row['image_predictions'], row['text_predictions']])
    return ' '.join(np.unique(x))


def get_image_predictions(df, embeddings, threshold=0.0):
    if len(df) > 3:
        KNN = 50
    else:
        KNN = 3

    model = NearestNeighbors(n_neighbors=KNN, metric='cosine')
    model.fit(embeddings)
    distances, indices = model.kneighbors(embeddings)

    predictions = []
    for k in tqdm(range(embeddings.shape[0])):
        idx = np.where(distances[k,] < threshold)[0]
        ids = indices[k, idx]
        posting_ids = df['posting_id'].iloc[ids].values
        if len(posting_ids) >= 2:
            idx_s = np.where(distances[k,] < threshold - 0.08888)[0]
            ids_s = indices[k, idx_s]
            posting_ids_b = df['posting_id'].iloc[ids_s].values
            if len(posting_ids_b) >= 2:
                predictions.append(posting_ids_b)
            else:
                predictions.append(posting_ids)
        else:
            idx = np.where(distances[k,] < 0.51313)[0]
            ids = indices[k, idx]
            posting_ids = df['posting_id'].iloc[ids].values
            predictions.append(posting_ids[:2])

    del model, distances, indices
    gc.collect()
    return predictions


def get_test_transforms():
    return A.Compose(
        [
            A.Resize(CFG.img_size, CFG.img_size, always_apply=True),
            A.Normalize(),
            ToTensorV2(p=1.0)
        ]
    )


class ShopeeDataset(Dataset):
    def __init__(self, image_paths, transforms=None):
        self.image_paths = image_paths
        self.augmentations = transforms

    def __len__(self):
        return self.image_paths.shape[0]

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']

        return image, torch.tensor(1)


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, scale=30.0, margin=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        return output


class ShopeeModel(nn.Module):

    def __init__(
            self,
            n_classes=CFG.classes,
            model_name=None,
            fc_dim=512,
            margin=CFG.margin,
            scale=CFG.scale,
            use_fc=True,
            pretrained=False):

        super(ShopeeModel, self).__init__()
        print('Building Model Backbone for {} model'.format(model_name))

        self.backbone = timm.create_model(model_name, pretrained=pretrained)

        if model_name == 'resnext50_32x4d':
            final_in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.backbone.global_pool = nn.Identity()

        elif 'efficientnet' in model_name:
            final_in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            self.backbone.global_pool = nn.Identity()

        elif model_name == 'eca_nfnet_l0':
            final_in_features = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Identity()
            self.backbone.head.global_pool = nn.Identity()

        self.pooling = nn.AdaptiveAvgPool2d(1)

        self.use_fc = use_fc

        self.dropout = nn.Dropout(p=0.0)
        self.fc = nn.Linear(final_in_features, fc_dim)
        self.bn = nn.BatchNorm1d(fc_dim)
        self._init_params()
        final_in_features = fc_dim

        self.final = ArcMarginProduct(
            final_in_features,
            n_classes,
            scale=scale,
            margin=margin,
            easy_margin=False,
            ls_eps=0.0
        )

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, image, label):
        feature = self.extract_feat(image)
        # logits = self.final(feature,label)
        return feature

    def extract_feat(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.pooling(x).view(batch_size, -1)

        if self.use_fc:
            x = self.dropout(x)
            x = self.fc(x)
            x = self.bn(x)
        return x


class Mish_func(torch.autograd.Function):
    """from: https://github.com/tyunist/memory_efficient_mish_swish/blob/master/mish.py"""

    @staticmethod
    def forward(ctx, i):
        result = i * torch.tanh(F.softplus(i))
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]

        v = 1. + i.exp()
        h = v.log()
        grad_gh = 1. / h.cosh().pow_(2)

        # Note that grad_hv * grad_vx = sigmoid(x)
        # grad_hv = 1./v
        # grad_vx = i.exp()

        grad_hx = i.sigmoid()

        grad_gx = grad_gh * grad_hx  # grad_hv * grad_vx

        grad_f = torch.tanh(F.softplus(i)) + i * grad_gx

        return grad_output * grad_f


class Mish(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        pass

    def forward(self, input_tensor):
        return Mish_func.apply(input_tensor)


def replace_activations(model, existing_layer, new_layer):
    """A function for replacing existing activation layers"""

    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            model._modules[name] = replace_activations(module, existing_layer, new_layer)

        if type(module) == existing_layer:
            layer_old = module
            layer_new = new_layer
            model._modules[name] = layer_new
    return model


def get_model(model_name=None, model_path=None, n_classes=None):
    model = ShopeeModel(model_name=model_name)
    if model_name == 'eca_nfnet_l0':
        model = replace_activations(model, torch.nn.SiLU, Mish())
    model.eval()
    model.load_state_dict(torch.load(model_path))
    model = model.to(CFG.device)

    return model


class EnsembleModel(nn.Module):

    def __init__(self):
        super(EnsembleModel, self).__init__()

        self.m1 = get_model('eca_nfnet_l0', 'input/shopee-pytorch-models/arcface_512x512_nfnet_l0 (mish).pt')
        self.m2 = get_model('tf_efficientnet_b5_ns', 'input/shopee-pytorch-models/arcface_512x512_eff_b5_.pt')

    def forward(self, img, label):
        feat1 = self.m1(img, label)
        feat2 = self.m2(img, label)

        return (feat1 + feat2) / 2


def get_image_embeddings(image_paths, model_name=None, model_path=None):
    embeds = []

    model = EnsembleModel()

    image_dataset = ShopeeDataset(image_paths=image_paths, transforms=get_test_transforms())
    image_loader = torch.utils.data.DataLoader(
        image_dataset,
        batch_size=CFG.batch_size,
        pin_memory=True,
        drop_last=False,
        num_workers=4
    )

    with torch.no_grad():
        for img, label in tqdm(image_loader):
            img = img.cuda()
            label = label.cuda()
            feat = model(img, label)
            image_embeddings = feat.detach().cpu().numpy()
            embeds.append(image_embeddings)

    del model
    image_embeddings = np.concatenate(embeds)
    print(f'Our image embeddings shape is {image_embeddings.shape}')
    del embeds
    gc.collect()
    return image_embeddings


def get_text_predictions(df, max_features=25_000):
    model = TfidfVectorizer(stop_words='english', binary=True, max_features=max_features)
    text_embeddings = model.fit_transform(df_cu['title']).toarray()
    preds = []
    CHUNK = 1024 * 4

    print('Finding similar titles...')
    CTS = len(df) // CHUNK
    if len(df) % CHUNK != 0: CTS += 1
    for j in range(CTS):

        a = j * CHUNK
        b = (j + 1) * CHUNK
        b = min(b, len(df))
        print('chunk', a, 'to', b)

        # COSINE SIMILARITY DISTANCE
        cts = cupy.matmul(text_embeddings, text_embeddings[a:b].T).T

        for k in range(b - a):
            # choose best threhod
            IDX = cupy.where(cts[k,] > 0.7705)[0]
            o = df_cu.iloc[cupy.asnumpy(IDX)].posting_id.to_pandas().values
            if len(o) >= 2:
                IDX_b = cupy.where(cts[k,] > 0.80105)[0]
                o_b = df_cu.iloc[cupy.asnumpy(IDX_b)].posting_id.to_pandas().values
                if len(o) >= 2:
                    preds.append(o_b)
                else:
                    preds.append(o)
            else:
                IDX = cupy.where(cts[k,] > 0.6555)[0]
                o = df_cu.iloc[cupy.asnumpy(IDX)].posting_id.to_pandas().values
                preds.append(o[:2])

    del model, text_embeddings
    gc.collect()
    return preds


df, df_cu, image_paths = read_dataset()

image_embeddings = get_image_embeddings(image_paths.values)
print('hello this image embeddings')

image_predictions = get_image_predictions(df, image_embeddings, threshold = 0.36)
text_predictions = get_text_predictions(df, max_features = 25_000)
print('hello this image+text embeddings')

df['image_predictions'] = image_predictions
df['text_predictions'] = text_predictions
df['matches'] = df.apply(combine_predictions, axis = 1)
df[['posting_id', 'matches']].to_csv('submission.csv', index = False)
###out put the result
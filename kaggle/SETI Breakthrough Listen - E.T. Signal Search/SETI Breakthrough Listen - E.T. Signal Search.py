#!pip install --upgrade fastai
#!pip install timm

import cv2
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from fastai.vision.all import *

set_seed(999, reproducible=True)
dataset_path = Path('../input/seti-breakthrough-listen')
df = pd.read_csv(dataset_path / 'train_labels.csv')
df['path'] = df['id'].apply(
    lambda x: str(dataset_path / 'train' / x[0] / x) + '.npy')  # adding the path for each id for easier processing


class SETIDataset:
    def __init__(self, df, spatial=True, sixchan=True):
        self.df = df
        self.spatial = spatial
        self.sixchan = sixchan

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        label = self.df.iloc[index].target
        filename = self.df.iloc[index].path
        data = np.load(filename).astype(np.float32)
        if not self.sixchan: data = data[::2].astype(np.float32)
        if self.spatial:
            data = np.vstack(data).transpose((1, 0))
            data = cv2.resize(data, dsize=(256, 256))
            data_tensor = torch.tensor(data).float().unsqueeze(0)
        else:
            data = np.transpose(data, (1, 2, 0))
            data = cv2.resize(data, dsize=(256, 256))
            data = np.transpose(data, (2, 0, 1)).astype(np.float32)
            data_tensor = torch.tensor(data).float()

        return (data_tensor, torch.tensor(label))


train_df, valid_df = train_test_split(df, test_size=0.2, random_state=999)
train_ds = SETIDataset(train_df)
valid_ds = SETIDataset(valid_df)

bs = 128
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=bs, num_workers=8)
valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=bs, num_workers=8)

dls = DataLoaders(train_dl, valid_dl)
from timm import create_model
from fastai.vision.learner import _update_first_layer


def create_timm_body(arch: str, pretrained=True, cut=None, n_in=3):
    "Creates a body from any model in the `timm` library."
    model = create_model(arch, pretrained=pretrained, num_classes=0, global_pool='')
    _update_first_layer(model, n_in, pretrained)
    if cut is None:
        ll = list(enumerate(model.children()))
        cut = next(i for i, o in reversed(ll) if has_pool_type(o))
    if isinstance(cut, int):
        return nn.Sequential(*list(model.children())[:cut])
    elif callable(cut):
        return cut(model)
    else:
        raise NamedError("cut must be either integer or function")


def create_timm_model(arch: str, n_out, cut=None, pretrained=True, n_in=3, init=nn.init.kaiming_normal_,
                      custom_head=None,
                      concat_pool=True, **kwargs):
    "Create custom architecture using `arch`, `n_in` and `n_out` from the `timm` library"
    body = create_timm_body(arch, pretrained, None, n_in)
    if custom_head is None:
        nf = num_features_model(nn.Sequential(*body.children()))
        head = create_head(nf, n_out, concat_pool=concat_pool, **kwargs)
    else:
        head = custom_head
    model = nn.Sequential(body, head)
    if init is not None: apply_init(model[1], init)
    return model


def timm_learner(dls, arch: str, loss_func=None, pretrained=True, cut=None, splitter=None,
                 y_range=None, config=None, n_in=3, n_out=None, normalize=True, **kwargs):
    "Build a convnet style learner from `dls` and `arch` using the `timm` library"
    if config is None: config = {}
    if n_out is None: n_out = get_c(dls)
    assert n_out, "`n_out` is not defined, and could not be inferred from data, set `dls.c` or pass `n_out`"
    if y_range is None and 'y_range' in config: y_range = config.pop('y_range')
    model = create_timm_model(arch, n_out, default_split, pretrained, n_in=n_in, y_range=y_range, **config)
    learn = Learner(dls, model, loss_func=loss_func, splitter=default_split, **kwargs)
    if pretrained: learn.freeze()
    return learn


def roc_auc(preds, targ):
    try:
        return roc_auc_score(targ.cpu(), preds.squeeze().cpu())
    except:
        return 0.5


learn = timm_learner(dls, 'resnext50_32x4d', pretrained=True, n_in=1, n_out=1, metrics=[roc_auc], opt_func=ranger,
                     loss_func=BCEWithLogitsLossFlat()).to_fp16()

learn.lr_find()

learn.fit_one_cycle(3, 0.1, cbs=[ReduceLROnPlateau()])

learn.recorder.plot_loss()

learn = learn.to_fp32()
learn.save('resnext50_32x4d-3epochs')
learn = learn.load('resnext50_32x4d-3epochs')
test_df = pd.read_csv(dataset_path/'sample_submission.csv')
test_df['path'] = test_df['id'].apply(lambda x: str(dataset_path/'test'/x[0]/x)+'.npy')
test_ds = SETIDataset(test_df)

bs = 128
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=bs, num_workers=8, shuffle=False)

preds = []
for xb, _ in tqdm(test_dl):
    with torch.no_grad(): output = learn.model(xb.cuda())
    preds.append(torch.sigmoid(output.float()).squeeze().cpu())
preds = torch.cat(preds)

sample_df = pd.read_csv(dataset_path/'sample_submission.csv')
sample_df['target'] = preds
sample_df.to_csv('submission.csv', index=False)
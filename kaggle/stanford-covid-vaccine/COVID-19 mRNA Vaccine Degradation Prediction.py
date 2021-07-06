from collections import OrderedDict
from fastprogress import progress_bar
from pathlib import Path
from sklearn.model_selection import train_test_split, ShuffleSplit
from torch import nn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

import functools
import os
import pandas as pd
import random
import shutil
import torch
import torch.nn.functional as F


target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
input_cols = ['sequence', 'structure', 'predicted_loop_type']
error_cols = ['reactivity_error', 'deg_error_Mg_pH10', 'deg_error_Mg_50C', 'deg_error_pH10', 'deg_error_50C']

token_dicts = {
    "sequence": {x: i for i, x in enumerate("ACGU")},
    "structure": {x: i for i, x in enumerate('().')},
    "predicted_loop_type": {x: i for i, x in enumerate("BEHIMSX")}
}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)




# Loader
from sklearn.model_selection import train_test_split, ShuffleSplit
from torch import nn
from torch.utils.data import Dataset

import functools


BASE_PATH = "/kaggle/input/stanford-covid-vaccine"
MODEL_SAVE_PATH = "/kaggle/model"


def preprocess_inputs(df, cols):
    return np.concatenate([preprocess_feature_col(df, col) for col in cols], axis=2)


def preprocess_feature_col(df, col):
    dic = token_dicts[col]
    dic_len = len(dic)
    seq_length = len(df[col][0])
    ident = np.identity(dic_len)
    # convert to one hot
    arr = np.array(
        df[[col]].applymap(lambda seq: [ident[dic[x]] for x in seq]).values.tolist()
    ).squeeze(1)
    # shape: data_size x seq_length x dic_length
    assert arr.shape == (len(df), seq_length, dic_len)
    return arr


def preprocess(base_data, is_test=False):
    inputs = preprocess_inputs(base_data, input_cols)
    if is_test:
        labels = None
    else:
        labels = np.array(base_data[target_cols].values.tolist()).transpose((0, 2, 1))
        assert labels.shape[2] == len(target_cols)
    assert inputs.shape[2] == 14
    return inputs, labels


def get_bpp_feature(bpp):
    bpp_nb_mean = 0.077522  # mean of bpps_nb across all training data
    bpp_nb_std = 0.08914  # std of bpps_nb across all training data
    bpp_max = bpp.max(-1)[0]
    bpp_sum = bpp.sum(-1)
    bpp_nb = torch.true_divide((bpp > 0).sum(dim=1), bpp.shape[1])
    bpp_nb = torch.true_divide(bpp_nb - bpp_nb_mean, bpp_nb_std)
    return [bpp_max.unsqueeze(2), bpp_sum.unsqueeze(2), bpp_nb.unsqueeze(2)]


@functools.lru_cache(5000)
def load_from_id(id_):
    path = Path(BASE_PATH) / f"bpps/{id_}.npy"
    data = np.load(str(path))
    return data


def get_distance_matrix(leng):
    idx = np.arange(leng)
    Ds = []
    for i in range(len(idx)):
        d = np.abs(idx[i] - idx)
        Ds.append(d)

    Ds = np.array(Ds) + 1
    Ds = 1 / Ds
    Ds = Ds[None, :, :]
    Ds = np.repeat(Ds, 1, axis=0)

    Dss = []
    for i in [1, 2, 4]:
        Dss.append(Ds ** i)
    Ds = np.stack(Dss, axis=3)
    print(Ds.shape)
    return Ds


def get_structure_adj(df):
    Ss = []
    for i in range(len(df)):
        seq_length = df["seq_length"].iloc[i]
        structure = df["structure"].iloc[i]
        sequence = df["sequence"].iloc[i]

        cue = []
        a_structures = OrderedDict([
            (("A", "U"), np.zeros([seq_length, seq_length])),
            (("C", "G"), np.zeros([seq_length, seq_length])),
            (("U", "G"), np.zeros([seq_length, seq_length])),
            (("U", "A"), np.zeros([seq_length, seq_length])),
            (("G", "C"), np.zeros([seq_length, seq_length])),
            (("G", "U"), np.zeros([seq_length, seq_length])),
        ])
        for j in range(seq_length):
            if structure[j] == "(":
                cue.append(j)
            elif structure[j] == ")":
                start = cue.pop()
                a_structures[(sequence[start], sequence[j])][start, j] = 1
                a_structures[(sequence[j], sequence[start])][j, start] = 1

        a_strc = np.stack([a for a in a_structures.values()], axis=2)
        a_strc = np.sum(a_strc, axis=2, keepdims=True)
        Ss.append(a_strc)

    Ss = np.array(Ss)
    return Ss


def create_loader(df, batch_size=1, is_test=False):
    features, labels = preprocess(df, is_test)
    features_tensor = torch.from_numpy(features)
    if labels is not None:
        labels_tensor = torch.from_numpy(labels)
        dataset = VacDataset(features_tensor, df, labels_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, drop_last=False)
    else:
        dataset = VacDataset(features_tensor, df, None)
        loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False, drop_last=False)
    return loader


class VacDataset(Dataset):
    def __init__(self, features, df, labels=None):
        self.features = features
        self.labels = labels
        self.test = labels is None
        self.ids = df["id"]
        self.score = None
        self.structure_adj = get_structure_adj(df)
        self.distance_matrix = get_distance_matrix(self.structure_adj.shape[1])
        if "score" in df.columns:
            self.score = df["score"]
        else:
            df["score"] = 1.0
            self.score = df["score"]
        self.signal_to_noise = None
        if not self.test:
            self.signal_to_noise = df["signal_to_noise"]
            assert self.features.shape[0] == self.labels.shape[0]
        else:
            assert self.ids is not None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        bpp = torch.from_numpy(load_from_id(self.ids[index]).copy()).float()
        adj = self.structure_adj[index]
        distance = self.distance_matrix[0]
        bpp = np.concatenate([bpp[:, :, None], adj, distance], axis=2)
        if self.test:
            return dict(sequence=self.features[index].float(), bpp=bpp, ids=self.ids[index])
        else:
            return dict(sequence=self.features[index].float(), bpp=bpp,
                        label=self.labels[index], ids=self.ids[index],
                        signal_to_noise=self.signal_to_noise[index],
                        score=self.score[index])





#Model
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import math


class Conv1dStack(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, padding=1, dilation=1):
        super(Conv1dStack, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
        )
        self.res = nn.Sequential(
            nn.Conv1d(out_dim, out_dim, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        h = self.res(x)
        return x + h


class Conv2dStack(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, padding=1, dilation=1):
        super(Conv2dStack, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
        )
        self.res = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        h = self.res(x)
        return x + h


class SeqEncoder(nn.Module):
    def __init__(self, in_dim: int):
        super(SeqEncoder, self).__init__()
        self.conv0 = Conv1dStack(in_dim, 128, 3, padding=1)
        self.conv1 = Conv1dStack(128, 64, 6, padding=5, dilation=2)
        self.conv2 = Conv1dStack(64, 32, 15, padding=7, dilation=1)
        self.conv3 = Conv1dStack(32, 32, 30, padding=29, dilation=2)

    def forward(self, x):
        x1 = self.conv0(x)
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        # x = x.permute(0, 2, 1).contiguous()
        # BATCH x 256 x seq_length
        return x


class BppAttn(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        super(BppAttn, self).__init__()
        self.conv0 = Conv1dStack(in_channel, out_channel, 3, padding=1)
        self.bpp_conv = Conv2dStack(5, out_channel)

    def forward(self, x, bpp):
        x = self.conv0(x)
        bpp = self.bpp_conv(bpp)
        # BATCH x C x SEQ x SEQ
        # BATCH x C x SEQ
        x = torch.matmul(bpp, x.unsqueeze(-1))
        return x.squeeze(-1)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerWrapper(nn.Module):
    def __init__(self, dmodel=256, nhead=8, num_layers=2):
        super(TransformerWrapper, self).__init__()
        self.pos_encoder = PositionalEncoding(256)
        encoder_layer = TransformerEncoderLayer(d_model=dmodel, nhead=nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        self.pos_emb = PositionalEncoding(dmodel)

    def flatten_parameters(self):
        pass

    def forward(self, x):
        x = x.permute((1, 0, 2)).contiguous()
        x = self.pos_emb(x)
        x = self.transformer_encoder(x)
        x = x.permute((1, 0, 2)).contiguous()
        return x, None


class RnnLayers(nn.Module):
    def __init__(self, dmodel, dropout=0.3, transformer_layers: int = 2):
        super(RnnLayers, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.rnn0 = TransformerWrapper(dmodel, nhead=8, num_layers=transformer_layers)
        self.rnn1 = nn.LSTM(dmodel, dmodel // 2, batch_first=True, num_layers=1, bidirectional=True)
        self.rnn2 = nn.GRU(dmodel, dmodel // 2, batch_first=True, num_layers=1, bidirectional=True)

    def forward(self, x):
        self.rnn0.flatten_parameters()
        x, _ = self.rnn0(x)
        if self.rnn1 is not None:
            self.rnn1.flatten_parameters()
            x = self.dropout(x)
            x, _ = self.rnn1(x)
        if self.rnn2 is not None:
            self.rnn2.flatten_parameters()
            x = self.dropout(x)
            x, _ = self.rnn2(x)
        return x

    
class BaseAttnModel(nn.Module):
    def __init__(self, transformer_layers: int = 2):
        super(BaseAttnModel, self).__init__()
        self.linear0 = nn.Linear(14 + 3, 1)
        self.seq_encoder_x = SeqEncoder(18)
        self.attn = BppAttn(256, 128)
        self.seq_encoder_bpp = SeqEncoder(128)
        self.seq = RnnLayers(256 * 2, dropout=0.3,
                             transformer_layers=transformer_layers)

    def forward(self, x, bpp):
        bpp_features = get_bpp_feature(bpp[:, :, :, 0].float())
        x = torch.cat([x] + bpp_features, dim=-1)
        learned = self.linear0(x)
        x = torch.cat([x, learned], dim=-1)
        x = x.permute(0, 2, 1).contiguous().float()
        # BATCH x 18 x seq_len
        bpp = bpp.permute([0, 3, 1, 2]).contiguous().float()
        # BATCH x 5 x seq_len x seq_len
        x = self.seq_encoder_x(x)
        # BATCH x 256 x seq_len
        bpp = self.attn(x, bpp)
        bpp = self.seq_encoder_bpp(bpp)
        # BATCH x 256 x seq_len
        x = x.permute(0, 2, 1).contiguous()
        # BATCH x seq_len x 256
        bpp = bpp.permute(0, 2, 1).contiguous()
        # BATCH x seq_len x 256
        x = torch.cat([x, bpp], dim=2)
        # BATCH x seq_len x 512
        x = self.seq(x)
        return x


class AEModel(nn.Module):
    def __init__(self, transformer_layers: int = 2):
        super(AEModel, self).__init__()
        self.seq = BaseAttnModel(transformer_layers=transformer_layers)
        self.linear = nn.Sequential(
            nn.Linear(256 * 2, 14),
            nn.Sigmoid(),
        )

    def forward(self, x, bpp):
        x = self.seq(x, bpp)
        x = F.dropout(x, p=0.3)
        x = self.linear(x)
        return x


class FromAeModel(nn.Module):
    def __init__(self, seq, pred_len=68, dmodel: int = 256):
        super(FromAeModel, self).__init__()
        self.seq = seq
        self.pred_len = pred_len
        self.linear = nn.Sequential(
            nn.Linear(dmodel * 2, len(target_cols)),
        )

    def forward(self, x, bpp):
        x = self.seq(x, bpp)
        x = self.linear(x)
        x = x[:, :self.pred_len]
        return x


#Pretrain
def learn_from_batch_ae(model, data, device):
    seq = data["sequence"].clone()
    seq[:, :, :14] = F.dropout2d(seq[:, :, :14], p=0.3)
    target = data["sequence"][:, :, :14]
    out = model(seq.to(device), data["bpp"].to(device))
    loss = F.binary_cross_entropy(out, target.to(device))
    return loss


def train_ae(model, train_data, optimizer, lr_scheduler, epochs=10, device="cpu",
             start_epoch: int = 0, start_it: int = 0, log_path: str = "./logs"):
    print(f"device: {device}")
    losses = []
    it = start_it
    model_save_path = Path(MODEL_SAVE_PATH)
    start_epoch = start_epoch
    end_epoch = start_epoch + epochs
    min_loss = 10.0
    min_loss_epoch = 0
    if not model_save_path.exists():
        model_save_path.mkdir(parents=True)
    for epoch in progress_bar(range(start_epoch, end_epoch)):
        print(f"epoch: {epoch}")
        model.train()
        for i, data in enumerate(train_data):
            optimizer.zero_grad()
            loss = learn_from_batch_ae(model, data, device)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            if lr_scheduler:
                lr_scheduler.step()
            loss_v = loss.item()
            losses.append(loss_v)
            it += 1
        loss_m = np.mean(losses)
        if loss_m < min_loss:
            min_loss_epoch = epoch
            min_loss = loss_m
        print(f'epoch: {epoch} loss: {loss_m}')
        losses = []
        torch.save(optimizer.state_dict(), str(model_save_path / "optimizer.pt"))
        torch.save(model.state_dict(), str(model_save_path / f"model-{epoch}.pt"))
    return dict(end_epoch=end_epoch, it=it, min_loss_epoch=min_loss_epoch)
In [7]:
import shutil


set_seed(123)
shutil.rmtree("./model", True)
shutil.rmtree("./logs", True)
save_path = Path("./model_prediction")
if not save_path.exists():
    save_path.mkdir(parents=True)

lr_scheduler = None
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AEModel()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
res = dict(end_epoch=0, it=0, min_loss_epoch=0)
epochs = [5, 5, 5, 5]
for e in epochs:
    res = train_ae(model, loader0, optimizer, lr_scheduler, e, device=device,
                   start_epoch=res["end_epoch"], start_it=res["it"])
    res = train_ae(model, loader1, optimizer, lr_scheduler, e, device=device,
                   start_epoch=res["end_epoch"], start_it=res["it"])
    res = train_ae(model, loader2, optimizer, lr_scheduler, e, device=device,
                   start_epoch=res["end_epoch"], start_it=res["it"])

epoch = res["min_loss_epoch"]
shutil.copyfile(str(Path(MODEL_SAVE_PATH) / f"model-{epoch}.pt"), "ae-model.pt")

#Training
device = torch.device('cuda')
BATCH_SIZE = 64
base_train_data = pd.read_json(str(Path(BASE_PATH) / 'train.json'), lines=True)
samples = base_train_data
save_path = Path("./model_prediction")
if not save_path.exists():
    save_path.mkdir(parents=True)
shutil.rmtree("./model", True)
shutil.rmtree("./logs", True)
split = ShuffleSplit(n_splits=5, test_size=.1)
ids = samples.reset_index()["id"]
set_seed(124)
for fold, (train_index, test_index) in enumerate(split.split(samples)):
    print(f"fold: {fold}")
    train_df = samples.loc[train_index].reset_index()
    val_df = samples.loc[test_index].reset_index()
    train_loader = create_loader(train_df, BATCH_SIZE)
    valid_loader = create_loader(val_df, BATCH_SIZE)
    print(train_df.shape, val_df.shape)
    ae_model = AEModel()
    state_dict = torch.load("./ae-model.pt")
    ae_model.load_state_dict(state_dict)
    del state_dict
    model = FromAeModel(ae_model.seq)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = None
    epoch = train(model, train_loader, valid_loader, optimizer, lr_scheduler, 200, device=device,
                  log_path=f"logs/{fold}")
    shutil.copyfile(str(Path(MODEL_SAVE_PATH) / f"./model-{epoch}.pt"), f"model_prediction/model-{fold}.pt")
    del model

def predict_batch(model, data, device):
    # batch x seq_len x target_size
    with torch.no_grad():
        pred = model(data["sequence"].to(device), data["bpp"].to(device))
        pred = pred.detach().cpu().numpy()
    return_values = []
    ids = data["ids"]
    for idx, p in enumerate(pred):
        id_ = ids[idx]
        assert p.shape == (model.pred_len, len(target_cols))
        for seqpos, val in enumerate(p):
            assert len(val) == len(target_cols)
            dic = {key: val for key, val in zip(target_cols, val)}
            dic["id_seqpos"] = f"{id_}_{seqpos}"
            return_values.append(dic)
    return return_values


def predict_data(model, loader, device, batch_size):
    data_list = []
    for i, data in enumerate(progress_bar(loader)):
        data_list += predict_batch(model, data, device)
    expected_length = model.pred_len * len(loader) * batch_size
    assert len(data_list) == expected_length, f"len = {len(data_list)} expected = {expected_length}"
    return data_list


#Prediction
device = torch.device('cuda') if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
base_test_data = pd.read_json(str(Path(BASE_PATH) / 'test.json'), lines=True)
public_df = base_test_data.query("seq_length == 107").copy()
private_df = base_test_data.query("seq_length == 130").copy()
print(f"public_df: {public_df.shape}")
print(f"private_df: {private_df.shape}")
public_df = public_df.reset_index()
private_df = private_df.reset_index()
pub_loader = create_loader(public_df, BATCH_SIZE, is_test=True)
pri_loader = create_loader(private_df, BATCH_SIZE, is_test=True)
pred_df_list = []
c = 0
for fold in range(5):
    model_load_path = f"./model_prediction/model-{fold}.pt"
    ae_model0 = AEModel()
    ae_model1 = AEModel()
    model_pub = FromAeModel(pred_len=107, seq=ae_model0.seq)
    model_pub = model_pub.to(device)
    model_pri = FromAeModel(pred_len=130, seq=ae_model1.seq)
    model_pri = model_pri.to(device)
    state_dict = torch.load(model_load_path, map_location=device)
    model_pub.load_state_dict(state_dict)
    model_pri.load_state_dict(state_dict)
    del state_dict

    data_list = []
    data_list += predict_data(model_pub, pub_loader, device, BATCH_SIZE)
    data_list += predict_data(model_pri, pri_loader, device, BATCH_SIZE)
    pred_df = pd.DataFrame(data_list, columns=["id_seqpos"] + target_cols)
    print(pred_df.head())
    print(pred_df.tail())
    pred_df_list.append(pred_df)
    c += 1
data_dic = dict(id_seqpos=pred_df_list[0]["id_seqpos"])
for col in target_cols:
    vals = np.zeros(pred_df_list[0][col].shape[0])
    for df in pred_df_list:
        vals += df[col].values
    data_dic[col] = vals / float(c)
pred_df_avg = pd.DataFrame(data_dic, columns=["id_seqpos"] + target_cols)
print(pred_df_avg.head())
pred_df_avg.to_csv("./submission.csv", index=False)


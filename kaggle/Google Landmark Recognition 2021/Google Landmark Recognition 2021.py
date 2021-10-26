import pathlib

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

import PIL.Image
import albumentations.pytorch
import cv2
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
from typing import List, Tuple

IMAGE_SIZE = 320  # (2021/09/24 5:00AM) Updated.
BATCH_SIZE = 512

MODEL_FILE = pathlib.Path('../input/google-landmark-2021-validation/model.pth')
TRAIN_LABEL_FILE = pathlib.Path('train.csv')
TRAIN_IMAGE_DIR = pathlib.Path('../input/landmark-recognition-2021/train')
VALID_LABEL_FILE = pathlib.Path('valid.csv')
VALID_IMAGE_DIR = pathlib.Path('../input/google-landmark-2021-validation/valid')
TEST_LABEL_FILE = pathlib.Path('../input/landmark-recognition-2021/sample_submission.csv')
TEST_IMAGE_DIR = pathlib.Path('../input/landmark-recognition-2021/test')
train_df = pd.read_csv('../input/landmark-recognition-2021/train.csv')

if len(train_df) == 1580470:
    records = {}

    for image_id, landmark_id in train_df.values:
        if landmark_id in records:
            records[landmark_id].append(image_id)
        else:
            records[landmark_id] = [image_id]

    image_ids = []
    landmark_ids = []

    for landmark_id, img_ids in records.items():
        num = min(len(img_ids), 2)
        image_ids.extend(records[landmark_id][:num])
        landmark_ids.extend([landmark_id] * num)

    train_df = pd.DataFrame({'id': image_ids, 'landmark_id': landmark_ids})

train_df.to_csv(TRAIN_LABEL_FILE, index=False)
valid_df = pd.read_csv('../input/google-landmark-2021-validation/valid.csv')
valid_df = valid_df[valid_df['landmark_id'] == -1]
valid_df.to_csv(VALID_LABEL_FILE, index=False)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, label_file: pathlib.Path, image_dir: pathlib.Path) -> None:
        super().__init__()
        self.files = [
            image_dir / n[0] / n[1] / n[2] / f'{n}.jpg'
            for n in pd.read_csv(label_file)['id'].values]

        self.transformer = albumentations.Compose([
            albumentations.SmallestMaxSize(IMAGE_SIZE, interpolation=cv2.INTER_CUBIC),
            albumentations.CenterCrop(IMAGE_SIZE, IMAGE_SIZE),
            albumentations.Normalize(),
            albumentations.pytorch.ToTensorV2(),
        ])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[str, torch.Tensor]:
        path = self.files[index]
        image = PIL.Image.open(self.files[index])
        image = self.transformer(image=np.array(image))['image']

        return path.name[:-4], image


@torch.no_grad()
def get_features(
        model: nn.Module,
        label_file: pathlib.Path,
        image_dir: pathlib.Path,
) -> Tuple[List[str], torch.Tensor]:
    loader = torch.utils.data.DataLoader(
        Dataset(label_file, image_dir),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = model.cuda()
    model.eval()

    all_names = []
    all_features = []

    for names, images in tqdm(loader, desc=image_dir.name):
        images = images.cuda()
        features = model(images)
        all_features.append(features)
        all_names.extend(names)

    return all_names, F.normalize(torch.cat(all_features, dim=0))


def get_similarity(model: nn.Module) -> Tuple[List[str], List[str]]:
    # features
    train_names, train_features = get_features(
        model, TRAIN_LABEL_FILE, TRAIN_IMAGE_DIR)
    _, valid_features = get_features(
        model, VALID_LABEL_FILE, VALID_IMAGE_DIR)
    test_names, test_features = get_features(
        model, TEST_LABEL_FILE, TEST_IMAGE_DIR)

    # penalties
    train_penalties_list = []
    for i in range(0, train_features.shape[0], 128):
        x = torch.mm(train_features[i:i + 128], valid_features.T)
        x = torch.topk(x, k=5)[0].mean(dim=1)
        train_penalties_list.append(x)
    train_penalties = torch.cat(train_penalties_list, dim=0)

    test_penalties_list = []
    for i in range(0, test_features.shape[0], 128):
        x = torch.mm(test_features[i:i + 128], valid_features.T)
        x = torch.topk(x, k=10)[0].mean(dim=1)
        test_penalties_list.append(x)
    test_penalties = torch.cat(test_penalties_list, dim=0)

    # neighbors
    submit_ids = []
    submit_landmark_ids = []
    submit_confidences = []

    train_df = pd.read_csv(TRAIN_LABEL_FILE)
    idmap = {n: v for n, v in train_df.values}

    for i in range(0, test_features.shape[0], 128):
        x = torch.mm(test_features[i:i + 128], train_features.T)
        x -= train_penalties[None, :]
        values, indexes = torch.topk(x, k=3)

        submit_ids.extend(test_names[i:i + 128])

        for idxs, vals, penalty in zip(indexes, values, test_penalties[i:i + 128]):
            scores = {}
            for idx, val in zip(idxs, vals):
                landmark_id = idmap[train_names[idx]]
                if landmark_id in scores:
                    scores[landmark_id] += float(val)
                else:
                    scores[landmark_id] = float(val)

            landmark_id, confidence = max(
                [(k, v) for k, v in scores.items()], key=lambda x: x[1])
            submit_landmark_ids.append(landmark_id)
            submit_confidences.append(confidence - penalty)

    # standardize confidence values
    max_conf = max(submit_confidences)
    min_conf = min(submit_confidences)
    submit_confidences = [
        (v - min_conf) / (max_conf - min_conf) for v in submit_confidences]

    # make values for 'landmark' column
    submit_landmarks = [
        f'{i} {c:.8f}' for i, c in zip(submit_landmark_ids, submit_confidences)]

    return submit_ids, submit_landmarks


model = torch.jit.load(str(MODEL_FILE))
print(model)
submit_ids, submit_landmarks = get_similarity(model)
submit_df = pd.DataFrame({'id': submit_ids, 'landmarks': submit_landmarks})
submit_df.to_csv('submission.csv', index=False)

submit_df = pd.read_csv('submission.csv')
submit_df['landmark_id'] = submit_df['landmarks'].apply(lambda x: int(x.split()[0]))
submit_df['confidence'] = submit_df['landmarks'].apply(lambda x: float(x.split()[1]))
train_df = pd.read_csv(TRAIN_LABEL_FILE)


def get_image(path, name):
    img = PIL.Image.open(path / name[0] / name[1] / name[2] / f'{name}.jpg')
    if img.width > img.height:
        img = img.resize((256, round(img.height / img.width * 256)))
        new_img = PIL.Image.new(img.mode, (256, 256), (0, 0, 0))
        new_img.paste(img, (0, (256 - img.height) // 2))
    else:
        img = img.resize((round(img.width / img.height * 256), 256))
        new_img = PIL.Image.new(img.mode, (256, 256), (0, 0, 0))
        new_img.paste(img, ((256 - img.width) // 2, 2))
    return np.array(new_img)


rows = 10
fig = plt.figure(figsize=(15, 4 * rows))
for r in range(rows):
    for c in range(3):
        i = r * 3 + c
        test_name, _, label, conf = submit_df.iloc[i].values
        test_image = get_image(TEST_IMAGE_DIR, test_name)
        train_name = train_df.query(f'landmark_id == {label}').iloc[0]['id']
        train_image = get_image(TRAIN_IMAGE_DIR, train_name)
        image = np.concatenate([test_image, train_image], axis=1)

        ax = fig.add_subplot(rows, 3, i + 1)
        ax.set_title(f'Label={label}, Confidence={conf:.2f}')
        ax.axis('off')
        ax.imshow(image)
fig.tight_layout()